mod maze;

use maze::{Field, Pos, NodeType};
use rand::{
    self, SeedableRng,
    distr::{
        Distribution,
        weighted::WeightedIndex,
    },
    rngs::StdRng,
    seq::IteratorRandom,
};
use std::collections::VecDeque;
use burn::{
    prelude::*,
    tensor::backend::AutodiffBackend,
    nn::{
        Linear, LinearConfig, Relu,
        loss::{
            HuberLoss, HuberLossConfig, Reduction,
        }
    },
    optim::{
        Adam, AdamConfig, GradientsParams, Optimizer,
        adaptor::OptimizerAdaptor,
    },
    backend::{
        Autodiff,
        libtorch::{LibTorch, LibTorchDevice},
    },
};
use strum::{IntoEnumIterator, EnumCount};
use itertools::MultiUnzip;

#[derive(Clone, Copy, PartialEq, Eq, Debug, strum::EnumCount, strum::EnumIter)]
enum Action {
    Up,
    Right,
    Down,
    Left,
}

impl Action {
    fn sample(dist: WeightedIndex<f32>) -> Self {
        let actions = Action::iter().collect::<Vec<_>>();
        let idx = dist.sample(&mut rand::rng());
        assert!(idx < Action::COUNT, "Invalid index for action");
        actions[idx]
    }
}

type State = Pos;

struct Env {
    field: Field,
    state: State,
    step: usize,
}

const MAX_STEP: usize = 20;
const MAX_EPISODE: usize = 20;
const ETA: f32 = 0.1;
const GAMMA: f32 = 0.9;
const EPSILON: f32 = 0.5;
const REPLAY_SIZE: usize = 4;


impl Env {
    fn new(field: Field) -> Self {
        Self {
            state: field.start,
            field: field,
            step: 0,
        }
    }

    fn step(&mut self, action: Action) -> (State, f32, bool) {
        let (dx, dy) = match action {
            Action::Up => (0, 1),
            Action::Down => (0, -1),
            Action::Left => (-1, 0),
            Action::Right => (1, 0),
        };
        let mut reward = 0.0;
        let pos = self.field.move_by(self.state, dx, dy);
        if pos != self.state {
            self.state = pos;
        } else {
            reward = -1.0;
        }

        if pos == self.field.goal {
            reward = 1.0;
        }

        self.step += 1;

        let mut done = false;
        if self.step >= MAX_STEP || reward > 0.0 {
            done = true;
        }

        (self.state, reward, done)
    }

    fn reset(&mut self) {
        self.state = self.field.start;
        self.step = 0;
    }
}


#[derive(Module, Debug)]
struct Model<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    activation: Relu,
}

impl<B: Backend> Model<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.activation.forward(self.linear1.forward(input));
        let x = self.activation.forward(self.linear2.forward(x));
        self.linear3.forward(x)
    }
}

#[derive(Config, Debug)]
struct ModelConfig {
    input_dim: usize,
    output_dim: usize,
    #[config(default = 16)]
    hidden_size: usize,
}

impl ModelConfig{
    fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let linear1 = LinearConfig::new(self.input_dim, self.hidden_size).init(device);
        let linear2 = LinearConfig::new(self.hidden_size, self.hidden_size).init(device);
        let linear3 = LinearConfig::new(self.hidden_size, self.output_dim).init(device);
        let activation = Relu::new();
        Model {
            linear1,
            linear2,
            linear3,
            activation,
        }
    }
}

struct Agent<B: AutodiffBackend> {
    model: Model<B>,
    optim: OptimizerAdaptor<Adam, Model<B>, B>,
    loss: HuberLoss,
    input_shape: (usize, usize),
    device: B::Device,
    memory: Memory,
}

impl<B: AutodiffBackend> Agent<B> {
    fn new(input_shape: (usize, usize), output_dim: usize, device: &B::Device) -> Self {
        let input_dim = input_shape.0 * input_shape.1;
        Self {
            model: ModelConfig::new(input_dim, output_dim).init(device),
            optim: AdamConfig::new().init(),
            loss: HuberLossConfig::new(1.0).init(),
            input_shape: input_shape,
            device: device.clone(),
            memory: Memory::new(42),
        }
    }
    fn decide(&self, state: State) -> Action {
        if rand::random::<f32>() < EPSILON {
            println!("Random action");
            let dist = WeightedIndex::new([0.25, 0.25, 0.25, 0.25]).unwrap();
            return Action::sample(dist);
        }

        let output = self.predict(&[state]);
        let idx: u8 = output.argmax(1).into_scalar().elem();
        Action::iter().collect::<Vec<_>>()[idx as usize]
    }

    fn learn(&mut self) {
        if self.memory.len() < REPLAY_SIZE * 10 {
            return;
        }

        let experiences = self.memory.pick_random();
        let (states, actions, rewards, state_nexts): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) = experiences.iter().map(|e| {
            (
                e.state,
                e.action as u8,
                e.reward,
                e.next_state,
            )
        }).multiunzip();

        let output = self.predict(&states);
        let action_tensor = Tensor::<B, 1, Int>::from_data(&*actions, &self.device).reshape([REPLAY_SIZE, 1]);
        let q = output.gather(1, action_tensor);
        // println!("output shape: {:?}", output.shape());
        // println!("q: {}", q.to_data());
        // println!("actions: {:?}", actions);
        // println!("tensor shape: {:?}", q.shape());

        let reward_tensor = Tensor::<B, 1>::from_data(&*rewards, &self.device).reshape([REPLAY_SIZE, 1]);
        let next_q_max = self.predict(&state_nexts).max_dim(1);
        let target = q.clone() + (next_q_max * GAMMA - q.clone() + reward_tensor) * ETA;
        // println!("reward shape: {:?}", reward_tensor.shape());
        // println!("q max shape: {:?}", next_q_max.shape());
        // println!("target shape: {:?}", target.shape());
        // println!("aaaaaaaaaaaaaaaa");
        // std::process::exit(0);

        let loss = self.loss.forward(q, target, Reduction::Mean);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        println!("Loss: {:.3}", loss.to_data());
        self.model = self.optim.step(0.01, self.model.clone(), grads);
    }

    fn build_input(&self, states: &[State]) -> Tensor<B, 2> {
        let input_dim = self.input_shape.0 * self.input_shape.1;
        let idxs = states.iter().map(|s| s.x + s.y * self.input_shape.0).collect::<Vec<_>>();
        let idx_tensor = Tensor::<B, 1>::from_data(&*idxs, &self.device);
        let tensor = idx_tensor.one_hot(input_dim);
        tensor
    }

    fn predict(&self, states: &[State]) -> Tensor<B, 2> {
        let input = self.build_input(states);
        self.model.forward(input)
    }

    fn dump_qvalue(&self) -> Tensor<B, 2> {
        let input = Tensor::eye(self.input_shape.0 * self.input_shape.1, &self.device);
        let output = self.model.forward(input);
        output
    }
}

struct Experience {
    state: State,
    action: Action,
    reward: f32,
    next_state: State,
}

struct Memory {
    storage: VecDeque<Experience>,
    rng: StdRng,
}

impl Memory {
    fn new(seed: u64) -> Self {
        Self {
            storage: VecDeque::new(),
            rng: SeedableRng::seed_from_u64(seed),
        }
    }

    fn len(&self) -> usize {
        self.storage.len()
    }

    fn add(&mut self, exp: Experience) {
        self.storage.push_back(exp);
    }

    fn pick_random(&mut self) -> Vec<&Experience> {
        self.storage.iter().choose_multiple(&mut self.rng, REPLAY_SIZE)
    }
}

fn print_qvalue<B: AutodiffBackend>(agent: &Agent<B>, env: &Env, field_sample: &[&str]) {
    let tensor = agent.dump_qvalue();
    for y in 0..env.field.height { 
        for x in 0..env.field.width { 
            let node = env.field.get(x, y).unwrap();
            if node.is_type(NodeType::Wall) {
                continue;
            }
            let q = tensor.clone().select(0, Tensor::from_data([x + y * env.field.width], &agent.device));
            println!("{}({}, {}): {:.3}", field_sample[y].chars().collect::<Vec<_>>()[x], y, x, q.to_data());
        }
    }
}

fn main() {
    let field_sample = [
        "#######",
        "#S....#",
        "##.#.##",
        "####.##",
        "#G....#",
        "#######",
    ];
    let field = Field::new(&field_sample);
    println!("{}", field);

    let mut env = Env::new(field);

    type B = Autodiff<LibTorch>;
    let device = LibTorchDevice::Mps;
    let mut agent = Agent::<B>::new((env.field.width, env.field.height), Action::COUNT, &device);

    for episode in 0..MAX_EPISODE {
        println!("--- start episode {} ---", episode);
        loop {
            let state = env.state;
            let action = agent.decide(state);
            let (state_next, reward, done) = env.step(action);
            println!("Step: {}, State: {}, Action: {:?}, StateN: {}, Reward: {}, Done: {}", env.step - 1, state, action, state_next, reward, done);

            agent.memory.add(Experience {
                state,
                action,
                reward,
                next_state: state_next,
            });
            agent.learn();
            if done {
                break;
            }
        }
        env.reset();

        println!("--- episode {} completed ---", episode);
        print_qvalue(&agent, &env, &field_sample);
    }
}
