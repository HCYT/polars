pub(crate) mod expr;
pub(crate) mod ir_graph;

use std::sync::Arc;

use ir_graph::{IRNodeEdgeKeys, build_ir_traversal_graph, unpack_edges_mut};
use polars_core::frame::UniqueKeepStrategy;
use polars_core::prelude::PlHashMap;
use polars_utils::arena::{Arena, Node};
use polars_utils::scratch_vec::ScratchVec;
use slotmap::{SlotMap, new_key_type};

use crate::dsl::{SinkTypeIR, UnionOptions};
use crate::plans::simplify_ordering::expr::{ExprOrderSimplifier, ObservableOrders};
use crate::plans::{IRAggExpr, is_scalar_ae};
use crate::prelude::{AExpr, IR};

#[derive(Default, Debug, Clone)]
enum Edge {
    #[default]
    Ordered,
    Unordered,
}

impl Edge {
    fn is_unordered(&self) -> bool {
        matches!(self, Self::Unordered)
    }
}

new_key_type! {
    struct EdgeKey;
}

type EdgesMap = SlotMap<EdgeKey, Edge>;

pub(crate) fn simplify_ir_ordering(
    roots: &[Node],
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) {
    let (mut ir_nodes_stack, mut ir_node_to_edges_map, mut all_edges_map) =
        build_ir_traversal_graph(roots, ir_arena);

    let eos_revisit_cache = &mut PlHashMap::default();
    let ae_nodes_scratch = &mut ScratchVec::default();
    let mut deleted_idxs = vec![];

    let mut simplifier = SimplifyIRNodeOrder {
        ir_node_to_edges_map: &mut ir_node_to_edges_map,
        all_edges_map: &mut all_edges_map,
        ir_arena,
        expr_arena,
        eos_revisit_cache,
        ae_nodes_scratch,
    };

    for (i, node) in ir_nodes_stack.iter().copied().enumerate() {
        if simplifier.simplify_ir_node_orders(node) {
            deleted_idxs.push(i)
        }
    }

    for (i, node) in ir_nodes_stack.drain(..).enumerate().rev() {
        if deleted_idxs.last() == Some(&i) {
            deleted_idxs.pop();
            continue;
        }

        simplifier.simplify_ir_node_orders(node);
    }
}

struct SimplifyIRNodeOrder<'a> {
    ir_node_to_edges_map: &'a mut PlHashMap<Node, IRNodeEdgeKeys<EdgeKey>>,
    all_edges_map: &'a mut EdgesMap,
    ir_arena: &'a mut Arena<IR>,
    expr_arena: &'a mut Arena<AExpr>,
    eos_revisit_cache: &'a mut PlHashMap<Node, ObservableOrders>,
    ae_nodes_scratch: &'a mut ScratchVec<Node>,
}

impl SimplifyIRNodeOrder<'_> {
    /// Returns if the node was deleted.
    fn simplify_ir_node_orders(&mut self, current_ir_node: Node) -> bool {
        use ObservableOrders as O;

        let current_ir_node_edges = self.ir_node_to_edges_map.get(&current_ir_node).unwrap();

        let IRNodeEdgeKeys {
            in_edges,
            out_edges,
            out_nodes: _,
        } = current_ir_node_edges;

        macro_rules! get_edge {
            ($edge_key:expr) => {
                self.all_edges_map.get($edge_key).unwrap()
            };
        }

        macro_rules! get_edge_mut {
            ($edge_key:expr) => {
                self.all_edges_map.get_mut($edge_key).unwrap()
            };
        }

        macro_rules! unpack_edges {
            ($total:literal) => {
                unpack_edges_mut::<EdgeKey, Edge, _, _, $total>(
                    current_ir_node_edges,
                    self.all_edges_map,
                )
                .unwrap()
            };
        }

        macro_rules! expr_order_simplifier {
            () => {{
                self.eos_revisit_cache.clear();
                ExprOrderSimplifier::new(self.expr_arena, self.eos_revisit_cache)
            }};
        }

        match self.ir_arena.get_mut(current_ir_node) {
            IR::Select { .. } | IR::HStack { .. } => {
                let (exprs, exprs_is_full_output) = match self.ir_arena.get_mut(current_ir_node) {
                    IR::Select { expr, .. } => (expr, true),
                    IR::HStack { exprs, schema, .. } => {
                        let v = schema.len() == exprs.len();
                        (exprs, v)
                    },
                    _ => unreachable!(),
                };

                let ([in_edge], [out_edge]) = unpack_edges!(2);

                let mut eos = expr_order_simplifier!();
                let ae_nodes_scratch = self.ae_nodes_scratch.get();

                ae_nodes_scratch.extend(exprs.iter().map(|eir| eir.node()));

                let exprs_observable_orders = eos.simplify_projected_exprs(
                    ae_nodes_scratch,
                    out_edge.is_unordered() && exprs_is_full_output,
                );

                let input_cols_order_observed_by_consumer =
                    !out_edge.is_unordered() && exprs_observable_orders.contains(O::COLUMN);

                // Mixed independent<>column order due to hstack
                let input_cols_order_observed_by_mixed_order_hstack =
                    !exprs_is_full_output && exprs_observable_orders.contains(O::INDEPENDENT);

                if !(input_cols_order_observed_by_consumer
                    || input_cols_order_observed_by_mixed_order_hstack
                    || eos.internally_observed_orders().contains(O::COLUMN))
                {
                    *in_edge = Edge::Unordered;
                }

                if exprs_observable_orders.is_empty()
                    || (exprs_observable_orders == O::COLUMN && in_edge.is_unordered())
                {
                    *out_edge = Edge::Unordered;
                }
            },

            IR::Sort {
                input,
                by_column,
                slice,
                sort_options,
            } => {
                let ([in_edge], [out_edge]) = unpack_edges!(2);

                if out_edge.is_unordered() && slice.is_none() {
                    *in_edge = out_edge.clone();
                    let input = *input;
                    return self.unlink_node(current_ir_node, input);
                }

                let mut eos = expr_order_simplifier!();
                let ae_nodes_scratch = self.ae_nodes_scratch.get();

                ae_nodes_scratch.extend(by_column.iter().map(|eir| eir.node()));

                let key_exprs_observable_orders =
                    eos.simplify_projected_exprs(ae_nodes_scratch, false);

                if in_edge.is_unordered()
                    || !(sort_options.maintain_order
                        || eos.internally_observed_orders().contains(O::COLUMN)
                        || key_exprs_observable_orders.contains(O::INDEPENDENT))
                {
                    *in_edge = Edge::Unordered;
                    sort_options.maintain_order = false;
                }
            },

            IR::Filter {
                input: _,
                predicate,
            } => {
                let ([in_edge], [out_edge]) = unpack_edges!(2);

                let mut eos = expr_order_simplifier!();
                let predicate_observable_orders =
                    eos.simplify_projected_exprs(&[predicate.node()], false);

                if out_edge.is_unordered()
                    && !(eos.internally_observed_orders().contains(O::COLUMN)
                        || predicate_observable_orders.contains(O::INDEPENDENT))
                {
                    *in_edge = Edge::Unordered;
                }

                if in_edge.is_unordered() {
                    *out_edge = Edge::Unordered;
                }
            },

            IR::GroupBy {
                input: _,
                keys,
                aggs,
                schema: _,
                maintain_order,
                options,
                apply,
            } => {
                let ([in_edge], [out_edge]) = unpack_edges!(2);

                // Put the implode in for the expr order optimizer.
                for agg in aggs.iter_mut() {
                    if !is_scalar_ae(agg.node(), self.expr_arena) {
                        agg.set_node(self.expr_arena.add(AExpr::Agg(IRAggExpr::Implode {
                            input: agg.node(),
                            maintain_order: true,
                        })));
                    }
                }

                let mut eos = expr_order_simplifier!();
                let ae_nodes_scratch = self.ae_nodes_scratch.get();

                ae_nodes_scratch.extend(keys.iter().map(|eir| eir.node()));
                let keys_observable = eos.simplify_projected_exprs(ae_nodes_scratch, false);

                ae_nodes_scratch.clear();
                ae_nodes_scratch.extend(aggs.iter().map(|eir| eir.node()));
                eos.simplify_projected_exprs(ae_nodes_scratch, false);

                let order_observing_options =
                    apply.is_some() || options.is_dynamic() || options.is_rolling();

                if !(order_observing_options
                    || keys_observable.contains(O::INDEPENDENT)
                    || eos.internally_observed_orders().contains(O::COLUMN)
                    || (*maintain_order && keys_observable.contains(O::COLUMN)))
                {
                    *maintain_order = false;
                    *in_edge = Edge::Unordered;
                }

                if in_edge.is_unordered() || !(*maintain_order || order_observing_options) {
                    *maintain_order = false;
                    *out_edge = Edge::Unordered;
                }
            },

            IR::Distinct { input: _, options } => {
                let ([in_edge], [out_edge]) = unpack_edges!(2);

                if !options.maintain_order || out_edge.is_unordered() {
                    options.maintain_order = false;
                    *out_edge = Edge::Unordered;
                }

                if in_edge.is_unordered()
                    || !(options.maintain_order
                        || matches!(
                            options.keep_strategy,
                            UniqueKeepStrategy::First | UniqueKeepStrategy::Last
                        ))
                {
                    options.maintain_order = false;
                    *in_edge = Edge::Unordered;
                }
            },

            IR::Join {
                input_left: _,
                input_right: _,
                schema: _,
                left_on,
                right_on,
                options,
            } => {
                let ([in_edge_lhs, in_edge_rhs], [out_edge]) = unpack_edges!(3);

                let mut eos = expr_order_simplifier!();

                let ae_nodes_scratch = self.ae_nodes_scratch.get();
                ae_nodes_scratch.extend(left_on.iter().map(|eir| eir.node()));
                let left_keys_observable = eos.simplify_projected_exprs(ae_nodes_scratch, false);

                ae_nodes_scratch.clear();
                ae_nodes_scratch.extend(right_on.iter().map(|eir| eir.node()));
                let right_keys_observable = eos.simplify_projected_exprs(ae_nodes_scratch, false);

                // Join keys should be elementwise.
                assert!(!(left_keys_observable | right_keys_observable).contains(O::INDEPENDENT));
                assert!(!eos.internally_observed_orders().contains(O::COLUMN));

                #[cfg(feature = "asof_join")]
                if let polars_ops::prelude::JoinType::AsOf(_) = &options.args.how {
                    return false;
                }

                use polars_ops::prelude::MaintainOrderJoin as JO;

                if out_edge.is_unordered() || options.args.maintain_order == JO::None {
                    *out_edge = Edge::Unordered;
                    *in_edge_lhs = Edge::Unordered;
                    *in_edge_rhs = Edge::Unordered;
                    Arc::make_mut(options).args.maintain_order = JO::None;
                }

                if in_edge_lhs.is_unordered() || options.args.maintain_order == JO::Right {
                    *in_edge_lhs = Edge::Unordered;

                    match options.args.maintain_order {
                        JO::Left => Arc::make_mut(options).args.maintain_order = JO::None,
                        JO::LeftRight | JO::RightLeft => {
                            Arc::make_mut(options).args.maintain_order = JO::Right
                        },
                        JO::None | JO::Right => {},
                    }
                }

                if in_edge_rhs.is_unordered() || options.args.maintain_order == JO::Left {
                    *in_edge_rhs = Edge::Unordered;

                    match options.args.maintain_order {
                        JO::Right => Arc::make_mut(options).args.maintain_order = JO::None,
                        JO::RightLeft | JO::LeftRight => {
                            Arc::make_mut(options).args.maintain_order = JO::Left
                        },
                        JO::None | JO::Left => {},
                    }
                }
            },

            IR::Union { inputs: _, options } => {
                assert_eq!(out_edges.len(), 1);

                let out_edge_key = *out_edges.first().unwrap();

                if !options.maintain_order || get_edge!(out_edge_key).is_unordered() {
                    options.maintain_order = false;
                    *get_edge_mut!(out_edge_key) = Edge::Unordered;
                    for k in in_edges.iter() {
                        *get_edge_mut!(*k) = Edge::Unordered;
                    }
                }

                // Note, having no ordered inputs still cannot de-order the out edge, since the rows
                // of each input are still ordered to fully appear before the next input.
            },

            #[cfg(feature = "merge_sorted")]
            IR::MergeSorted {
                input_left,
                input_right,
                key: _,
            } => {
                let ([in_edge_lhs, in_edge_rhs], [out_edge]) = unpack_edges!(3);

                if out_edge.is_unordered()
                    || (in_edge_lhs.is_unordered() && in_edge_rhs.is_unordered())
                {
                    *out_edge = Edge::Unordered;
                    *in_edge_lhs = Edge::Unordered;
                    *in_edge_rhs = Edge::Unordered;

                    let input_left = *input_left;
                    let input_right = *input_right;

                    self.ir_arena.replace(
                        current_ir_node,
                        IR::Union {
                            inputs: vec![input_left, input_right],
                            options: UnionOptions {
                                maintain_order: false,
                                ..Default::default()
                            },
                        },
                    );
                }
            },

            IR::MapFunction { input: _, function } => {
                let ([in_edge], [out_edge]) = unpack_edges!(2);

                if !function.observes_input_order()
                    && (!function.has_equal_order() || out_edge.is_unordered())
                {
                    *in_edge = Edge::Unordered;
                }

                if !function.is_order_producing(!in_edge.is_unordered())
                    && (in_edge.is_unordered() || !function.has_equal_order())
                {
                    *out_edge = Edge::Unordered;
                }
            },

            IR::HConcat { .. }
            | IR::SimpleProjection { .. }
            | IR::Slice { .. }
            | IR::ExtContext { .. } => {
                if in_edges.iter().all(|k| get_edge!(*k).is_unordered()) {
                    for k in out_edges.iter() {
                        *get_edge_mut!(*k) = Edge::Unordered
                    }
                }
            },

            IR::Cache { .. } => {
                if out_edges.iter().all(|k| get_edge!(*k).is_unordered()) {
                    for k in in_edges.iter() {
                        *get_edge_mut!(*k) = Edge::Unordered
                    }
                } else if in_edges.iter().all(|k| get_edge!(*k).is_unordered()) {
                    for k in out_edges.iter() {
                        *get_edge_mut!(*k) = Edge::Unordered
                    }
                }
            },

            IR::Sink { input: _, payload } => {
                let ([in_edge], []) = unpack_edges!(1);

                if let SinkTypeIR::Partitioned(options) = payload {
                    let mut eos = expr_order_simplifier!();
                    let ae_nodes_scratch = self.ae_nodes_scratch.get();

                    ae_nodes_scratch.extend(options.expr_irs_iter().map(|eir| eir.node()));
                    let observable = eos.simplify_projected_exprs(ae_nodes_scratch, false);

                    // Partition key exprs should be elementwise
                    assert!(!observable.contains(O::INDEPENDENT));
                    assert!(!eos.internally_observed_orders().contains(O::COLUMN));
                }

                if !payload.maintain_order() || in_edge.is_unordered() {
                    *in_edge = Edge::Unordered;
                    payload.set_maintain_order(false);
                }
            },

            #[cfg(feature = "python")]
            IR::PythonScan { .. } => {},

            IR::Scan { .. } | IR::DataFrameScan { .. } => {},

            IR::SinkMultiple { .. } | IR::Invalid => unreachable!(),
        };

        false
    }

    fn unlink_node(&mut self, current_ir_node: Node, input_to_current_ir_node: Node) -> bool {
        let current_ir_node_edges = self.ir_node_to_edges_map.get(&current_ir_node).unwrap();

        let IRNodeEdgeKeys {
            out_nodes,
            in_edges,
            ..
        } = current_ir_node_edges;

        assert_eq!(out_nodes.len(), 1);
        assert_eq!(in_edges.len(), 1);

        let current_in_edge_key = in_edges[0];

        let consumer_node = out_nodes[0];

        let mut iter = self
            .ir_arena
            .get_mut(consumer_node)
            .inputs_mut()
            .enumerate()
            .filter(|(_, node)| **node == current_ir_node);

        let (consumer_node_input_idx, node) = iter.next().unwrap();
        *node = input_to_current_ir_node;
        assert!(iter.next().is_none());

        let [
            Some(IRNodeEdgeKeys {
                in_edges: consumer_node_in_edges,
                ..
            }),
            Some(IRNodeEdgeKeys {
                out_edges: out_edges_of_new_input_node,
                out_nodes: new_input_node_out_nodes,
                ..
            }),
        ] = self
            .ir_node_to_edges_map
            .get_disjoint_mut([&consumer_node, &input_to_current_ir_node])
        else {
            unreachable!()
        };

        let out_edge_idx_in_new_input_node = out_edges_of_new_input_node
            .iter()
            .position(|k| *k == current_in_edge_key)
            .unwrap();

        out_edges_of_new_input_node[out_edge_idx_in_new_input_node] =
            consumer_node_in_edges[consumer_node_input_idx];
        new_input_node_out_nodes[out_edge_idx_in_new_input_node] = consumer_node;

        true
    }
}
