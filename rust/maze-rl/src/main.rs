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

use eframe::{
    egui, 
    epaint::{PathStroke, StrokeKind},
};
use rand::Rng;

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

const MAX_STEP: usize = 50;
const MAX_EPISODE: usize = 20;
const ETA: f32 = 0.1;
const GAMMA: f32 = 0.9;
const EPSILON: f32 = 0.5;
const BATCH_SIZE: usize = 8;
const REPLAY_BUFFER_MAX: usize = 100;


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
    target_model: Model<B>,
    optim: OptimizerAdaptor<Adam, Model<B>, B>,
    loss: HuberLoss,
    input_shape: (usize, usize),
    device: B::Device,
    memory: Memory,
}

impl<B: AutodiffBackend> Agent<B> {
    fn new(input_shape: (usize, usize), output_dim: usize, device: &B::Device) -> Self {
        let input_dim = input_shape.0 * input_shape.1;
        let model = ModelConfig::new(input_dim, output_dim).init(&device.clone());
        Self {
            target_model: model.clone().no_grad(),
            model,
            optim: AdamConfig::new().init(),
            loss: HuberLossConfig::new(1.0).init(),
            input_shape,
            device: device.clone(),
            memory: Memory::new(42),
        }
    }

    fn update_target_model(&mut self) {
        self.target_model = self.model.clone().no_grad();
    }

    fn decide(&self, state: State) -> Action {
        if rand::random::<f32>() < EPSILON {
            println!("Random action");
            let dist = WeightedIndex::new([0.25, 0.25, 0.25, 0.25]).unwrap();
            return Action::sample(dist);
        }

        let output = self.predict(&[state], true);
        let idx: u8 = output.argmax(1).into_scalar().elem();
        Action::iter().collect::<Vec<_>>()[idx as usize]
    }

    fn learn(&mut self) {
        if self.memory.len() < BATCH_SIZE * 10 {
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

        let output = self.predict(&states, false);
        let action_tensor = Tensor::<B, 1, Int>::from_data(&*actions, &self.device).reshape([BATCH_SIZE, 1]);
        let q = output.gather(1, action_tensor);
        // println!("output shape: {:?}", output.shape());
        // println!("q: {}", q.to_data());
        // println!("actions: {:?}", actions);
        // println!("tensor shape: {:?}", q.shape());

        let reward_tensor = Tensor::<B, 1>::from_data(&*rewards, &self.device).reshape([BATCH_SIZE, 1]);
        let next_q_max = self.predict(&state_nexts, true).max_dim(1);
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
        self.model = self.optim.step(0.05, self.model.clone(), grads);
    }

    fn build_input(&self, states: &[State]) -> Tensor<B, 2> {
        let input_dim = self.input_shape.0 * self.input_shape.1;
        let idxs = states.iter().map(|s| s.x + s.y * self.input_shape.0).collect::<Vec<_>>();
        let idx_tensor = Tensor::<B, 1>::from_data(&*idxs, &self.device);
        let tensor = idx_tensor.one_hot(input_dim);
        tensor
    }

    fn predict(&self, states: &[State], is_target: bool) -> Tensor<B, 2> {
        let input = self.build_input(states);
        if is_target {
            return self.target_model.forward(input);
        }
        self.model.forward(input)
    }

    fn memorize(&mut self, state: State, action: Action, reward: f32, next_state: State) {
        if self.memory.len() >= REPLAY_BUFFER_MAX {
            self.memory.storage.pop_front();
        }
        self.memory.add(Experience {
            state,
            action,
            reward,
            next_state,
        });
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
        self.storage.iter().choose_multiple(&mut self.rng, BATCH_SIZE)
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
            println!("{}({}, {}): {:.3}", field_sample[y].chars().collect::<Vec<_>>()[x], x, y, q.to_data());
        }
    }
}

const GRID_WIDTH: usize = 5;
const GRID_HEIGHT: usize = 5;

#[derive(Default)]
struct QVisualizer {
    q_values: [[[f32; 4]; GRID_WIDTH]; GRID_HEIGHT],
}

impl eframe::App for QVisualizer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let (response, painter) = ui.allocate_painter(
                egui::Vec2::new(400.0, 400.0),
                egui::Sense::hover(),
            );

            let rect = response.rect;
            let cell_w = rect.width() / GRID_WIDTH as f32;
            let cell_h = rect.height() / GRID_HEIGHT as f32;

            for y in 0..GRID_HEIGHT {
                for x in 0..GRID_WIDTH {
                    let x0 = rect.left() + x as f32 * cell_w;
                    let y0 = rect.top() + y as f32 * cell_h;
                    let x1 = x0 + cell_w;
                    let y1 = y0 + cell_h;
                    let cx = (x0 + x1) / 2.0;
                    let cy = (y0 + y1) / 2.0;

                    let q = self.q_values[y][x];
                    let min_q = q.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max_q = q.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let normalize = |v: f32| (v - min_q) / (max_q - min_q + 1e-5);
                    let to_color = |v: f32| {
                        let n = normalize(v);
                        egui::Color32::from_rgb(
                            (n * 255.0) as u8,
                            0,
                            ((1.0 - n) * 255.0) as u8,
                        )
                    };
                    let to_stroke = |v: f32| {
                        PathStroke::new(1.0, to_color(v))
                    };

                    let corners = [
                        egui::pos2(x0, y0),         // top-left
                        egui::pos2(x1, y0),         // top-right
                        egui::pos2(x1, y1),         // bottom-right
                        egui::pos2(x0, y1),         // bottom-left
                        egui::pos2(cx, cy),         // center
                    ];

                    // up
                    painter.add(egui::Shape::convex_polygon(
                        vec![corners[0], corners[1], corners[4]],
                        to_color(q[0]),
                        to_stroke(q[0]),
                    ));
                    // right
                    painter.add(egui::Shape::convex_polygon(
                        vec![corners[1], corners[2], corners[4]],
                        to_color(q[1]),
                        to_stroke(q[1]),
                    ));
                    // down
                    painter.add(egui::Shape::convex_polygon(
                        vec![corners[2], corners[3], corners[4]],
                        to_color(q[2]),
                        to_stroke(q[2]),
                    ));
                    // left
                    painter.add(egui::Shape::convex_polygon(
                        vec![corners[3], corners[0], corners[4]],
                        to_color(q[3]),
                        to_stroke(q[3]),
                    ));

                    // optional: grid border
                    painter.rect_filled(
                        egui::Rect::from_min_max(
                            egui::pos2(x0, y0),
                            egui::pos2(x1, y1),
                        ),
                        0.0,
                        egui::Color32::TRANSPARENT,
                    );
                    painter.rect_stroke(
                        egui::Rect::from_min_max(
                            egui::pos2(x0, y0),
                            egui::pos2(x1, y1),
                        ),
                        0.0,
                        egui::Stroke::new(1.0, egui::Color32::GRAY),
                        StrokeKind::Middle,
                    );
                }
            }

            // Q値をランダムに更新（ダミー学習）
            let mut rng = rand::thread_rng();
            for y in 0..GRID_HEIGHT {
                for x in 0..GRID_WIDTH {
                    for a in 0..4 {
                        self.q_values[y][x][a] += rng.gen_range(-0.01..0.01);
                    }
                }
            }

            ctx.request_repaint(); // 毎フレーム更新
        });
    }
}

fn main() -> eframe::Result<()> {
    let app = QVisualizer::default();
    let native_options = eframe::NativeOptions::default();
    eframe::run_native("visualizer", native_options, Box::new(|_cc| Ok(Box::new(app))))
}

// fn main() {
//     let field_sample = [
//         "#######",
//         "#S....#",
//         "##.#.##",
//         "####.##",
//         "#G....#",
//         "#######",
//     ];
//     let field = Field::new(&field_sample);
//     println!("{}", field);
//
//     let mut env = Env::new(field);
//
//     type B = Autodiff<LibTorch>;
//     let device = LibTorchDevice::Mps;
//     let mut agent = Agent::<B>::new((env.field.width, env.field.height), Action::COUNT, &device);
//
//     for episode in 0..MAX_EPISODE {
//         println!("--- start episode {} ---", episode);
//         loop {
//             let state = env.state;
//             let action = agent.decide(state);
//             let (state_next, reward, done) = env.step(action);
//             println!("Step: {}, State: {}, Action: {:?}, StateN: {}, Reward: {}, Done: {}", env.step - 1, state, action, state_next, reward, done);
//
//             agent.memorize(state, action, reward, state_next);
//             agent.learn();
//
//             if done {
//                 break;
//             }
//         }
//         agent.update_target_model();
//         env.reset();
//
//         println!("--- episode {} completed ---", episode);
//         print_qvalue(&agent, &env, &field_sample);
//     }
// }
