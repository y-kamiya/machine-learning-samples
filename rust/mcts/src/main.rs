use rand::prelude::*;
use rand::seq::IndexedRandom;
use maze::{Action, Field, Pos, NodeType};

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

fn ucb1(tree: &Tree, index: usize) -> f64 {
    let node = tree.get(index);
    assert!(node.parent != None);

    let parent = tree.get(node.parent.unwrap());
    let n = node.n_visit as f64;
    let n_all = parent.n_visit as f64;

    let exploit = node.score / n;
    let explore = (2.0 * n_all.ln() / n).sqrt();
    // println!("i:{}, exploit:{}, explore:{}", index, exploit, explore);
    
    exploit + explore
}

fn find_best_child(tree: &Tree, children: &Vec<usize>) -> usize {
    let mut rng = rand::rng();
    let ucb1_max = children.iter().map(|&i| ucb1(tree, i)).fold(f64::NEG_INFINITY, f64::max);
    let best_index = children.iter()
        .filter(|i| ucb1(tree, **i) == ucb1_max)
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

    assert_ne!(next_state, None);
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

const MAX_ROLLOUT: usize = 100;

fn rollout(tree: &Tree, index: usize, field: &Field) -> f64 {
    let mut rng = rand::rng();
    let node = tree.get(index);
    if field.is_goal(node.state) {
        return 1.0;
    }

    for _ in 0..MAX_ROLLOUT {
        let movables = field.movable_actions(node.state);
        let action = movables.iter().choose(&mut rng).unwrap();
        let to = field.act(node.state, *action);
        if field.is_goal(to) {
            return 1.0;
        }
    }
    return 0.0;
}

fn update_node(node: &mut TreeNode, reward: f64) {
    node.n_visit += 1;
    node.score = reward / node.n_visit as f64;
}

fn update(tree: &mut Tree, index: usize, field: &Field, reward: f64) {
    let mut idx = Some(index);
    while idx != None {
        let node = tree.get_mut(idx.unwrap());
        update_node(node, reward);
        idx = node.parent;
    }
}

const N_TRAIN: usize = 100;

fn mcts() {
    let _field = create_field(true);

    let mut _tree = Tree::new();
    _tree.add_node(TreeNode::new(_field.start));

    for i in 0..N_TRAIN {
        println!("\ntrain {}", i);

        let index = select(&_tree, &_field);
        let node = _tree.get(index);
        println!("selected: {:?}", node);

        let new_index = expand(&mut _tree, index, &_field);
        println!("expanded index: {:?}", new_index);

        let reward = rollout(&_tree, new_index, &_field);
        println!("rollout reward: {:?}", reward);

        update(&mut _tree, new_index, &_field, reward);
    }


}

fn main() {
    // random_maze();
    mcts();
}
