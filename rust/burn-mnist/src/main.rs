use burn::{
    prelude::*,
    tensor::backend::AutodiffBackend,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
        Linear, LinearConfig, Relu, Dropout, DropoutConfig, PaddingConfig2d,
        loss::CrossEntropyLossConfig,
    },
    data::{
        dataloader::{
            DataLoaderBuilder,
            batcher::Batcher,
        },
        dataset::vision::{MnistDataset, MnistItem},
    },
    optim::SgdConfig,
    record::CompactRecorder,
    train::{
        TrainStep, TrainOutput, ClassificationOutput, ValidStep, LearnerBuilder,
        metric::{
            AccuracyMetric, LossMetric,
        },
    },
    backend::{
        Autodiff,
        // ndarray::{NdArray, NdArrayDevice},
        libtorch::{LibTorch, LibTorchDevice},
    },
};

#[derive(Module, Debug)]
struct Cnn<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
    pool: MaxPool2d,
    dropout: Dropout,
}

impl<B: Backend> Cnn<B> {
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let [bs, h, w] = input.dims();
        let x = input.reshape([bs, 1, h, w]);

        let x = self.activation.forward(self.conv1.forward(x));
        let x = self.pool.forward(x);
        let x = self.activation.forward(self.conv2.forward(x));
        let x = self.pool.forward(x);
        let x = x.reshape([bs, 7 * 7 * 64]);
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        self.linear2.forward(x)
    }

    fn forward_classification(&self, input: Tensor<B, 3>, labels: Tensor<B, 1, Int>) -> ClassificationOutput<B> {
        let logits = self.forward(input);
        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), labels.clone());

        ClassificationOutput::new(loss, logits, labels)
    }
}

impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for Cnn<B> {
    fn step(&self, batch: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.labels);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for Cnn<B> {
    fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.labels)
    }
}

#[derive(Config, Debug)]
struct CnnConfig {
    #[config(default = 10)]
    num_classes: usize,
    #[config(default = 256)]
    hidden_size: usize,
    #[config(default = 0.4)]
    dropout: f64,
}

impl CnnConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Cnn<B> {
        let pad = PaddingConfig2d::Explicit(2, 2);
        Cnn {
            conv1: Conv2dConfig::new([1, 32], [5, 5]).with_padding(pad.clone()).init(device),
            conv2: Conv2dConfig::new([32, 64], [5, 5]).with_padding(pad).init(device),
            linear1: LinearConfig::new(7 * 7 * 64, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            activation: Relu::new(),
            pool: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}


#[derive(Clone, Default)]
struct MnistBatcher {}

#[derive(Clone, Debug)]
struct MnistBatch<B: Backend> {
    images: Tensor<B, 3>,
    labels: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, MnistItem, MnistBatch<B>> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> MnistBatch<B> {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 2>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            .map(|tensor| ((tensor - 255) - 0.1307) / 0.3081)
            .collect();
        let labels = items
            .iter()
            .map(|item| {
                let label = (item.label as i64).elem::<B::IntElem>();
                Tensor::<B, 1, Int>::from_data([label], device)
            })
            .collect();

        let images = Tensor::cat(images, 0);
        let labels = Tensor::cat(labels, 0);

        MnistBatch { images,  labels }
    }
}


#[derive(Config, Debug)]
struct TrainConfig {
    #[config(default = 2)]
    n_epochs: usize,
    #[config(default = 4)]
    n_workers: usize,
    #[config(default = 4)]
    batch_size: usize,
    #[config(default = 42)]
    seed: u64,
    #[config(default = 0.01)]
    lr: f64,
}

fn create_output_dir(dir: &str) {
    std::fs::remove_dir_all(dir).ok();
    std::fs::create_dir_all(dir).ok();
}

fn train<B: AutodiffBackend>(config: TrainConfig, device: B::Device) {
    let output_dir = "output";
    create_output_dir(output_dir);
    config.save(format!("{output_dir}/config.json"))
          .expect("Failed to save config");

    B::seed(config.seed);

    let model: Cnn<B> = CnnConfig::new().init(&device);
    let optimizer = SgdConfig::new().init();

    let batcher = MnistBatcher::default();

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.n_workers)
        .build(MnistDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .num_workers(config.n_workers)
        .build(MnistDataset::test());

    let learner = LearnerBuilder::new(output_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.n_epochs)
        .summary()
        .build(model, optimizer, config.lr);

    let model_trained = learner.fit(dataloader_train, dataloader_test);
    model_trained
        .save_file(format!("{output_dir}/model"), &CompactRecorder::new())
        .expect("Failed to save model");
}

fn main() {
    // type B = Autodiff<NdArray>;
    // let device = NdArrayDevice::Cpu;
    type B = Autodiff<LibTorch>;
    let device = LibTorchDevice::Mps;
    let config = TrainConfig::new();
    train::<B>(config, device);
}
