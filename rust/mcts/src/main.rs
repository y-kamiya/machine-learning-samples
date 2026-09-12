use rand::prelude::*;
use rand::{SeedableRng, rngs::StdRng};
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

    fn update_node(&mut self, reward: f64) {
        self.n_visit += 1;
        self.score += reward;
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
        debug_assert!(index < self.nodes.len());
        &self.nodes[index]
    }

    fn get_mut(&mut self, index: usize) -> &mut TreeNode {
        debug_assert!(index < self.nodes.len());
        &mut self.nodes[index]
    }

    fn ucb1(&self, index: usize) -> f64 {
        let node = self.get(index);
        debug_assert!(node.parent != None);

        let parent = self.get(node.parent.unwrap());
        let n = node.n_visit as f64;
        let n_all = parent.n_visit as f64;

        let exploit = node.score / n;
        let explore = (2.0 * n_all.ln() / n).sqrt();
        log::trace!("i:{}, exploit:{:.4}, explore:{:.4}", index, exploit, explore);
        
        exploit + explore
    }

    fn n_visit(&self, index: usize) -> f64 {
        let node = self.get(index);
        debug_assert!(node.parent != None);
        node.n_visit as f64
    }

    fn best_children<F>(&self, parent: usize, func: F) -> Vec<&usize> 
    where F: Fn(&Tree, usize) -> f64,
    {
        let node = self.get(parent);
        let ucb1_max = node.children.iter().map(|&i| func(self, i)).fold(f64::NEG_INFINITY, f64::max);
        let best_indexes = node.children.iter()
            .filter(|i| func(self, **i) == ucb1_max).collect();
        best_indexes
    }

    fn update(&mut self, index: usize, reward: f64) {
        let mut idx = Some(index);
        while idx != None {
            let node = self.get_mut(idx.unwrap());
            node.update_node(reward);
            idx = node.parent;
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Config {
    seed: u64,
    n_rollout: usize,
    n_train: usize,
}

struct MCTS {
    cfg: Config,
    rng: StdRng,
    tree: Tree,
    field: Field,
}

impl MCTS {
    fn new(cfg: Config) -> MCTS {
        let field = create_field(true);

        let mut tree = Tree::new();
        tree.add_node(TreeNode::new(field.start));

        MCTS {
            cfg: cfg,
            rng: StdRng::seed_from_u64(cfg.seed),
            tree: tree,
            field: field,
        }
    }

    fn actions_not_expanded(&self, index: usize) -> Vec<Action> {
        let mut not_expanded = Vec::new();

        let node = self.tree.get(index);
        let movables = self.field.movable_actions(node.state);
        for action in movables {
            let to = self.field.act(node.state, action);
            let has_state = node.children.iter().any(|i| self.tree.get(*i).state == to);
            if !has_state {
                not_expanded.push(action);
            }
        }
        not_expanded
    }

    fn is_fully_expanded(&self, index: usize) -> bool {
        if self.tree.get(index).children.is_empty() {
            return false;
        }

        let not_expanded = self.actions_not_expanded(index);
        if not_expanded.is_empty() {
            return true;
        }
        return false;
    }

    fn select(&mut self) -> usize {
        let mut index = 0;
        while self.is_fully_expanded(index) {
            let indexes = self.tree.best_children(index, Tree::ucb1);
            index = **indexes.choose(&mut self.rng).unwrap();
        }
        index
    }

    fn create_new_node(&self, parent_index: usize) -> TreeNode {
        let parent = self.tree.get(parent_index);
        let movables = self.field.movable_actions(parent.state);
        let mut next_state = None;
        for action in movables {
            let pos = self.field.act(parent.state, action);
            let has_node = parent.children.iter().any(|i| self.tree.get(*i).state == pos);
            if !has_node {
                next_state = Some(pos);
                break;
            }
        }

        debug_assert_ne!(next_state, None);
        let mut node = TreeNode::new(next_state.unwrap());
        node.parent = Some(parent_index);
        node
    }

    fn expand(&mut self, parent_index: usize) -> usize {
        let new_node = self.create_new_node(parent_index);
        let new_index = self.tree.add_node(new_node);
        let parent = self.tree.get_mut(parent_index);
        parent.add_child(new_index)
    }

    fn rollout(&mut self, index: usize) -> f64 {
        let node = self.tree.get(index);
        if self.field.is_goal(node.state) {
            return 1.0;
        }

        let mut pos = node.state;
        for i in 0..self.cfg.n_rollout {
            let movables = self.field.movable_actions(pos);
            let action = movables.iter().choose(&mut self.rng).unwrap();
            let to = self.field.act(pos, *action);
            if self.field.is_goal(to) {
                return 1.0 / (i as f64 + 1.0);
            }
            pos = to;
            
        }
        return 0.0;
    }

    fn train(&mut self) {
        for i in 0..self.cfg.n_train {
            log::debug!("\ntrain {}", i);

            let index = self.select();
            let node = self.tree.get(index);
            log::debug!("selected: {:?}", node);

            let new_index = self.expand(index);
            log::debug!("expanded index: {:?}", new_index);

            let reward = self.rollout(new_index);
            log::debug!("rollout reward: {:?}", reward);

            self.tree.update(new_index, reward);
        }
    }

    fn execute(&mut self) {
        log::info!("\n--------------------------------");
        log::info!("start execute()");

        let mut index: usize = 0;
        for _ in 0..10 {
            let node = self.tree.get(index);
            log::info!("{:?}", node);

            if self.field.is_goal(node.state) {
                log::info!("end execute(): reached goal");
                break;
            }

            let indexes = self.tree.best_children(index, Tree::n_visit);
            if indexes.is_empty() {
                log::info!("end execute(): no child");
            }
            index = **indexes.choose(&mut self.rng).unwrap();
        }
    }
}

fn main() {
    env_logger::init();

    let cfg = Config {
        seed: 42,
        n_rollout: 5,
        n_train: 50,
    };
    let mut mcts = MCTS::new(cfg);
    mcts.train();
    mcts.execute();
}
