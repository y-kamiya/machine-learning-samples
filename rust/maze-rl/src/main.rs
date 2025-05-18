mod maze;

use maze::{Field, Pos, NodeType};
use rand;
use rand::distr::{
    Distribution,
    weighted::WeightedIndex,
};
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

#[derive(Clone, Copy, PartialEq, Eq, Debug, strum::EnumCount, strum::EnumIter)]
enum Action {
    Up,
    Down,
    Left,
    Right,
}

impl Action {
    fn sample(dist: WeightedIndex<f32>) -> Self {
        let actions = Action::iter().collect::<Vec<_>>();
        let idx = dist.sample(&mut rand::rng());
        assert!(idx < Action::COUNT, "Invalid index for action");
        actions[idx]
    }
}

struct Env {
    field: Field,
    state: Pos,
    step: usize,
}

const MAX_STEP: usize = 10;
const MAX_EPISODE: usize = 1;

impl Env {
    fn new(field: Field) -> Self {
        Self {
            state: field.start,
            field: field,
            step: 0,
        }
    }

    fn step(&mut self, action: Action) -> (Pos, f32, bool) {
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

        if self.field.field.get(&pos).unwrap().is_type(NodeType::Goal) {
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
    #[config(default = 32)]
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
}

const ETA: f32 = 0.1;
const GAMMA: f32 = 0.9;
const EPSILON: f32 = 0.5;

impl<B: AutodiffBackend> Agent<B> {
    fn new(input_shape: (usize, usize), output_dim: usize, device: &B::Device) -> Self {
        let input_dim = input_shape.0 * input_shape.1;
        Self {
            model: ModelConfig::new(input_dim, output_dim).init(device),
            optim: AdamConfig::new().init(),
            loss: HuberLossConfig::new(1.0).init(),
            input_shape: input_shape,
            device: device.clone(),
        }
    }
    fn decide(&self, state: Pos) -> Action {
        if rand::random::<f32>() < EPSILON {
            println!("Random action");
            let dist = WeightedIndex::new([0.25, 0.25, 0.25, 0.25]).unwrap();
            return Action::sample(dist);
        }

        let output = self.predict(state, false);
        let idx: u8 = output.argmax(1).into_scalar().elem();
        Action::iter().collect::<Vec<_>>()[idx as usize]
    }

    fn learn(&mut self, state: Pos, state_next: Pos, action: Action, reward: f32) {
        let output = self.predict(state, true);
        let q = output.select(0, Tensor::from_data([action as usize], &self.device));

        let target = if reward > 0.0 {
            q.clone() + (-q.clone() + reward) * ETA
        } else {
            let next_q_max = self.predict(state_next, false).max_dim(1);
            q.clone() + (next_q_max * GAMMA - q.clone()) * ETA
        };

        let loss = self.loss.forward(q, target, Reduction::Mean);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        self.model = self.optim.step(0.01, self.model.clone(), grads);
    }

    fn build_input(&self, state: Pos, require_grad: bool) -> Tensor<B, 2> {
        let input_dim = self.input_shape.0 * self.input_shape.1;
        let mut array = vec![0.0; input_dim];
        let idx = state.x + state.y * self.input_shape.0;
        array[idx] = 1.0;
        let tensor = Tensor::<B, 1>::from_floats(&*array, &self.device).reshape([1, input_dim]);
        tensor.set_require_grad(require_grad)
    }

    fn predict(&self, state: Pos, require_grad: bool) -> Tensor<B, 2> {
        let input = self.build_input(state, require_grad);
        self.model.forward(input)
    }

    fn dump_qvalue(&self) -> Tensor<B, 2> {
        let input = Tensor::eye(self.input_shape.0 * self.input_shape.1, &self.device);
        let output = self.model.forward(input);
        output
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
        loop {
            let state = env.state;
            let action = agent.decide(state);
            let (state_next, reward, done) = env.step(action);
            println!("Episode: {}, Step: {}, State: {}, StateN: {}, Reward: {}, Done: {}", episode, env.step - 1, state, state_next, reward, done);

            agent.learn(state, state_next, action, reward);
            if done {
                break;
            }
        }
        env.reset();
    }
    println!("--- completed ---");
    let tensor = agent.dump_qvalue();
    for y in 0..env.field.height { 
        for x in 0..env.field.width { 
            let node = env.field.get(x, y).unwrap();
            if node.is_type(NodeType::Wall) {
                continue;
            }
            let q = tensor.clone().select(0, Tensor::from_data([x + y * env.field.width], &device));
            println!("{}({}, {}): {:.3}", field_sample[y].chars().collect::<Vec<_>>()[x], y, x, q.to_data());
        }
    }
}
