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
        Linear, LinearConfig, Relu, Dropout, DropoutConfig, PaddingConfig2d,
        loss::HuberLossConfig,
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

const MAX_STEP: usize = 5;
const MAX_EPISODE: usize = 3;

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
        let pos = self.field.move_by(self.state, dx, dy);
        if pos != self.state {
            self.state = pos;
        }

        let mut reward = 0.0;
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
    fn forward(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
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

struct Agent<B: Backend> {
    model: Model<B>,
    input_shape: (usize, usize),
    device: B::Device,
}

impl<B: AutodiffBackend> Agent<B> {
    fn new(input_shape: (usize, usize), output_dim: usize, device: &B::Device) -> Self {
        Self {
            model: ModelConfig::new(input_dim, output_dim).init(device),
            input_shape: input_shape,
            device: device.clone(),
        }
    }
    fn decide(&self) -> Action {
        let dist = WeightedIndex::new([0.25, 0.25, 0.25, 0.25]).unwrap();
        Action::sample(dist)
    }

    fn learn(&self, state: Pos, state_next: Pos, action: Action, reward: f32) {
        let input = self.build_input(state);
        let output = self.model.forward(input);

        let loss = HuberLossConfig::new()
            .init(&self.device)
            .forward(output.clone(), Tensor::<B, 1>::from_floats(vec![reward], &self.device));
    }

    fn build_input(&self, state: Pos) -> Tensor<B, 1> {
        let mut array = vec![0.0; self.input_shape.0 * self.input_shape.1];
        array[state.y * self.input_shape.0 + state.x] = 1.0;
        Tensor::<B, 1>::from_floats(array, &self.device)
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
    let agent = Agent::<B>::new((env.field.width, env.field.height), Action::COUNT, &device);

    for episode in 0..MAX_EPISODE {
        loop {
            let state = env.state;
            let action = agent.decide();
            let (state_next, reward, done) = env.step(action);
            println!("Episode: {}, Step: {}, State: {}, StateN: {}, Reward: {}, Done: {}", episode, env.step - 1, state, state_next, reward, done);

            agent.learn(state, state_next, action, reward);
            if done {
                break;
            }
        }
        env.reset();
    }
}
