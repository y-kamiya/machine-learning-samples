use rand::prelude::*;
use rand::seq::IndexedRandom;
use maze::{Action, Field, Pos};

use mcts::create_field;

type State = Pos;

#[derive(Debug)]
struct TreeNode {
    parent: Option<usize>,
    children: Vec<usize>,
    n_visit: u32,
    score: f64,
    state: State,
}

impl TreeNode {
    fn new(state: State) -> TreeNode {
        TreeNode {
            parent: None,
            children: Vec::new(),
            n_visit: 0,
            score: 0.0,
            state: state,
        }
    }

    fn add_child(&mut self, index: usize) -> usize {
        self.children.push(index);
        index
    }
}

struct Tree {
    nodes: Vec<TreeNode>,
}

impl Tree {
    fn new() -> Tree {
        Tree { nodes: Vec::new() }
    }

    fn add_node(&mut self, node: TreeNode) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    fn root(&self) -> &TreeNode {
        &self.nodes[0]
    }

    fn get(&self, index: usize) -> &TreeNode {
        assert!(index < self.nodes.len());
        &self.nodes[index]
    }

    fn get_mut(&mut self, index: usize) -> &mut TreeNode {
        assert!(index < self.nodes.len());
        &mut self.nodes[index]
    }
}

fn actions_not_expanded(tree: &Tree, index: usize, field: &Field) -> Vec<Action> {
    let mut not_expanded = Vec::new();

    let node = tree.get(index);
    let movables = field.movable_actions(node.state);
    for action in movables {
        let to = field.act(node.state, action);
        let has_state = node.children.iter().any(|i| tree.get(*i).state == to);
        if !has_state {
            not_expanded.push(action);
        }
    }
    not_expanded
}

fn is_fully_expanded(tree: &Tree, index: usize, field: &Field) -> bool {
    if tree.get(index).children.is_empty() {
        return false;
    }

    let not_expanded = actions_not_expanded(tree, index, field);
    if not_expanded.is_empty() {
        return true;
    }
    return false;
}

fn find_best_child(tree: &Tree, children: &Vec<usize>) -> usize {
    let mut rng = rand::rng();
    let score_max = children.iter().map(|&i| tree.get(i).score).fold(f64::NEG_INFINITY, f64::max);
    let best_index = children.iter()
        .filter(|i| tree.get(**i).score == score_max)
        .choose(&mut rng).unwrap();
    *best_index
}

fn select(tree: &Tree, field: &Field) -> usize {
    let mut index = 0;
    while is_fully_expanded(tree, index, field) {
        let node = tree.get(index);
        index = find_best_child(&tree, &node.children);
    }
    index
}

fn create_new_node(tree: &Tree, parent_index: usize, field: &Field) -> TreeNode {
    let parent = tree.get(parent_index);
    let movables = field.movable_actions(parent.state);
    let mut next_state = None;
    for action in movables {
        let pos = field.act(parent.state, action);
        let has_node = parent.children.iter().any(|i| tree.get(*i).state == pos);
        if !has_node {
            next_state = Some(pos);
            break;
        }
    }

    assert_eq!(next_state, None);
    let mut node = TreeNode::new(next_state.unwrap());
    node.parent = Some(parent_index);
    node
}

fn expand(tree: &mut Tree, parent_index: usize, field: &Field) -> usize {
    let new_node = create_new_node(tree, parent_index, field);
    let new_index = tree.add_node(new_node);
    let parent = tree.get_mut(parent_index);
    parent.add_child(new_index)
}

fn rollout() {
}

fn update() {
}

fn mcts() {
    let _field = create_field(true);

    let mut _tree = Tree::new();
    _tree.add_node(TreeNode::new(_field.start));

    for _ in 0..1 {
        let index = select(&_tree, &_field);
        let node = _tree.get(index);
        println!("{:?}", node);

        let new_index = expand(&mut _tree, index, &_field);
        println!("{:?}", new_index);
    }


}

fn main() {
    // random_maze();
    mcts();
}
