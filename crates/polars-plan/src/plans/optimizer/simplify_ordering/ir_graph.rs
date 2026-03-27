use polars_core::prelude::{InitHashMaps, PlHashMap};
use polars_utils::UnitVec;
use polars_utils::arena::{Arena, Node};
use polars_utils::unique_id::UniqueId;
use slotmap::SlotMap;

use crate::prelude::IR;

#[derive(Default, Debug)]
pub(crate) struct IRNodeEdgeKeys<EdgeKey> {
    pub(crate) in_edges: UnitVec<EdgeKey>,
    pub(crate) out_edges: UnitVec<EdgeKey>,
    pub(crate) out_nodes: UnitVec<Node>,
}

/// Builds an IR traversal graph where caches are visited only after all of their consumers are
/// visited.
#[expect(clippy::type_complexity)]
pub(crate) fn build_ir_traversal_graph<EdgeKey, Edge>(
    roots: &[Node],
    ir_arena: &mut Arena<IR>,
) -> (
    Vec<Node>,                                // Nodes in sink->source traversal order
    PlHashMap<Node, IRNodeEdgeKeys<EdgeKey>>, // Edge keys for each node
    SlotMap<EdgeKey, Edge>,                   // Edges slotmap
)
where
    EdgeKey: slotmap::Key,
    Edge: Default,
{
    let mut cache_hits: PlHashMap<UniqueId, usize> = PlHashMap::new();
    let mut num_nodes: usize = 0;

    let mut ir_nodes_stack = Vec::with_capacity(roots.len() + 8);
    ir_nodes_stack.extend_from_slice(roots);

    while let Some(ir_node) = ir_nodes_stack.pop() {
        let ir = ir_arena.get(ir_node);

        if let IR::Cache { id, .. } = ir {
            let _ = cache_hits.try_insert(*id, 0);
            *cache_hits.get_mut(id).unwrap() += 1;
        } else {
            num_nodes += 1;
        }

        ir.copy_inputs(&mut ir_nodes_stack);
    }

    num_nodes += cache_hits.len();

    let mut all_edges_map: SlotMap<EdgeKey, Edge> = SlotMap::with_capacity_and_key(num_nodes);
    let mut ir_node_to_edges_map: PlHashMap<Node, IRNodeEdgeKeys<EdgeKey>> =
        PlHashMap::with_capacity(num_nodes);

    ir_nodes_stack.reserve_exact(num_nodes);
    ir_nodes_stack.extend_from_slice(roots);

    for i in 0..num_nodes + 1 {
        let Some(current_node) = ir_nodes_stack.get(i).copied() else {
            break;
        };

        assert!(i < num_nodes);

        let ir = ir_arena.get(current_node);

        if let IR::Cache { id, .. } = ir {
            let hits = cache_hits.get_mut(id).unwrap();
            *hits -= 1;

            if *hits != 0 {
                debug_assert!(i < ir_nodes_stack.len());
                continue;
            }
        }

        let inputs_start_idx = ir_nodes_stack.len();
        ir_arena.get(current_node).copy_inputs(&mut ir_nodes_stack);
        let num_inputs = ir_nodes_stack.len() - inputs_start_idx;

        let current_node_in_edges =
            UnitVec::from_iter((0..num_inputs).map(|_| all_edges_map.insert(Edge::default())));

        for i in 0..num_inputs {
            let input_node = ir_nodes_stack[i + inputs_start_idx];
            let _ = ir_node_to_edges_map.try_insert(input_node, IRNodeEdgeKeys::default());
            let IRNodeEdgeKeys {
                out_edges: input_node_out_edges,
                out_nodes: input_node_out_nodes,
                ..
            } = ir_node_to_edges_map.get_mut(&input_node).unwrap();

            input_node_out_edges.push(current_node_in_edges[i]);
            input_node_out_nodes.push(current_node);
        }

        let _ = ir_node_to_edges_map.try_insert(current_node, IRNodeEdgeKeys::default());
        let current_edges = ir_node_to_edges_map.get_mut(&current_node).unwrap();

        assert!(current_edges.in_edges.is_empty());
        current_edges.in_edges = current_node_in_edges;
    }

    (ir_nodes_stack, ir_node_to_edges_map, all_edges_map)
}

pub(crate) fn unpack_edges_mut<
    'a,
    EdgeKey: slotmap::Key,
    Edge,
    const NUM_INPUTS: usize,
    const NUM_OUTPUTS: usize,
    // Workaround for generic_const_exprs, have the caller pass in `NUM_INPUTS + NUM_OUTPUTS`
    const TOTAL_EDGES: usize,
>(
    node_edge_keys: &IRNodeEdgeKeys<EdgeKey>,
    edges_map: &'a mut SlotMap<EdgeKey, Edge>,
) -> Option<([&'a mut Edge; NUM_INPUTS], [&'a mut Edge; NUM_OUTPUTS])> {
    const {
        assert!(NUM_INPUTS + NUM_OUTPUTS == TOTAL_EDGES);
    }

    let in_: [_; NUM_INPUTS] = node_edge_keys.in_edges.as_slice().try_into().ok()?;
    let out: [_; NUM_OUTPUTS] = node_edge_keys.out_edges.as_slice().try_into().ok()?;

    let combined: [EdgeKey; TOTAL_EDGES] = std::array::from_fn(|i| {
        if i < NUM_INPUTS {
            in_[i]
        } else {
            out[i - NUM_INPUTS]
        }
    });

    let mut combined_mut_refs: [Option<&mut Edge>; TOTAL_EDGES] =
        edges_map.get_disjoint_mut(combined).unwrap().map(Some);

    let in_mut_refs: [&'a mut Edge; NUM_INPUTS] =
        std::array::from_fn(|i| combined_mut_refs[i].take().unwrap());

    let out_mut_refs: [&'a mut Edge; NUM_OUTPUTS] =
        std::array::from_fn(|i| combined_mut_refs[i + NUM_INPUTS].take().unwrap());

    Some((in_mut_refs, out_mut_refs))
}
