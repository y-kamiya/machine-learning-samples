use std::collections::HashMap;
use std::fmt;
use strum::{IntoEnumIterator, EnumCount};
use rand::{
    distr::{
        Distribution,
        weighted::WeightedIndex,
    },
};

#[derive(Clone, Copy, PartialEq, Eq, Debug, strum::EnumCount, strum::EnumIter, strum::Display)]
pub enum Action {
    Up,
    Right,
    Down,
    Left,
}

impl Action {
    pub fn uniform() -> Self {
        let dist = WeightedIndex::new([0.25, 0.25, 0.25, 0.25]).unwrap();
        Action::sample(dist)
    }

    pub fn sample(dist: WeightedIndex<f32>) -> Self {
        let actions = Action::iter().collect::<Vec<_>>();
        let idx = dist.sample(&mut rand::rng());
        assert!(idx < Action::COUNT, "Invalid index for action");
        actions[idx]
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Pos {
    pub x: usize,
    pub y: usize,
}

impl fmt::Display for Pos {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl Pos {
    fn new(x: usize, y: usize) -> Self {
        Pos { x, y }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NodeType {
    Start,
    Goal,
    Wall,
    Road,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Node {
    node_type: NodeType,
    pos: Pos,
}

impl Node {
    pub fn is_type(&self, node_type: NodeType) -> bool {
        self.node_type == node_type
    }
}

#[derive(Clone, Debug)]
pub struct Field {
    pub field: HashMap<Pos, Node>,
    pub start: Pos,
    pub goal: Pos,
    pub width: usize,
    pub height: usize,
}

impl Field {
    pub fn new(data: &[&str]) -> Field {
        let mut map = HashMap::new();
        let mut start: Option<Pos> = None;
        let mut goal: Option<Pos> = None;

        for (y, row) in data.iter().enumerate() {
            for (x, c) in row.chars().enumerate() {
                let node_type = match c {
                    'S' => NodeType::Start,
                    'G' => NodeType::Goal,
                    '#' => NodeType::Wall,
                    _ => NodeType::Road,
                };
                let pos = Pos::new(x, y);
                if node_type == NodeType::Start {
                    start = Some(pos);
                }
                if node_type == NodeType::Goal {
                    goal = Some(pos);
                }
                map.insert(
                    pos,
                    Node {
                        node_type: node_type,
                        pos: pos,
                    },
                );
            }
        }
        assert!(start.is_some(), "S not found");
        assert!(goal.is_some(), "G not found");

        let h = data.len();
        assert!(h > 1, "data must have at least 2 rows");
        let w = data[0].len();
        assert!(w > 1, "data must have at least 2 cols");

        Self {
            field: map,
            start: start.unwrap(),
            goal: goal.unwrap(),
            width: w,
            height: h,
        }
    }

    pub fn act(&self, from: Pos, action: Action) -> Pos {
        let (dx, dy) = match action {
            Action::Up => (0, 1),
            Action::Down => (0, -1),
            Action::Left => (-1, 0),
            Action::Right => (1, 0),
        };
        self.move_by(from, dx, dy)
    }

    pub fn move_by(&self, from: Pos, dx: i32, dy: i32) -> Pos {
        let x = from.x as i32 + dx;
        let y = from.y as i32 + dy;
        if x < 0 || self.width as i32 <= x || y < 0 || self.height as i32 <= y {
            return from;
        }

        let pos = Pos::new(x as usize, y as usize);
        let node = self.field.get(&pos).unwrap();
        if node.node_type == NodeType::Wall {
            return from;
        }

        pos
    }

    pub fn get(&self, x: usize, y: usize) -> Option<&Node> {
        self.field.get(&Pos::new(x, y))
    }

    pub fn movable_actions(&self, from: Pos) -> Vec<Action> {
        let mut movable_actions = Vec::new();
        for action in Action::iter() {
            let to = self.act(from, action);
            if from == to {
                continue;
            }
            movable_actions.push(action);
        }

        movable_actions
    }
}

impl fmt::Display for Field {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for y in 0..self.height {
            for x in 0..self.width {
                let pos = Pos::new(x, y);
                let node = self.field.get(&pos).unwrap();
                let c = match node.node_type {
                    NodeType::Start => 'S',
                    NodeType::Goal => 'G',
                    NodeType::Wall => '#',
                    NodeType::Road => '.',
                };
                write!(f, "{}", c)?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}
