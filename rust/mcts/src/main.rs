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
}

struct Tree {
    nodes: Vec<TreeNode>,
}

impl Tree {
    fn new() -> Tree {
        Tree { nodes: Vec::new() }
    }

    fn add_node(&mut self, node: TreeNode) {
        self.nodes.push(node);
    }

    fn root(&self) -> &TreeNode {
        &self.nodes[0]
    }

    fn get(&self, index: usize) -> &TreeNode {
        assert!(0 <= index && index < self.nodes.len());
        &self.nodes[index]
    }

    // fn find_best_leaf(&self) -> &TreeNode {
    //     let mut rng = rand::rng();
    //     let mut node = self.root();
    //     while !node.children.is_empty() {
    //         let score_max = node.children.iter().map(|&n| self.nodes[n].score).fold(f64::NEG_INFINITY, f64::max);
    //         let best_index = node.children.iter()
    //             .filter(|n| self.nodes[**n].score == score_max)
    //             .choose(&mut rng).unwrap();
    //         node = &self.nodes[*best_index];
    //     }
    //
    //     &node
    // }
}

fn actions_not_expanded(tree: &Tree, index: usize, field: &Field) -> Vec<Action> {
    let mut not_expanded = Vec::new();

    let node = tree.get(index);
    let movables = field.movable_actions(node.state);
    for action in movables {
        let to = field.act(node.state, action);
        let has_state = node.children.iter().any(|n| tree.get(*n).state == to);
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
    let score_max = children.iter().map(|&n| tree.get(n).score).fold(f64::NEG_INFINITY, f64::max);
    let best_index = children.iter()
        .filter(|n| tree.get(**n).score == score_max)
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

// fn expand(tree: &Tree, parent: usize) -> &TreeNode {
//     // TreeNodeの新規作成
//     // treeに追加
//     // parentのchildrenにindex追加
// }

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
    }


}

fn main() {
    // random_maze();
    mcts();
}
