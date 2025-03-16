from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, List

from keras import Model
from keras.api import datasets
from sqlalchemy import create_engine, Integer, String, ForeignKey, UniqueConstraint
from sqlalchemy.ext.orderinglist import ordering_list
from sqlalchemy.orm import relationship, sessionmaker, Mapped, mapped_column, DeclarativeBase

from jkcsoft.ml.log_utils import logger

log = logger(__name__)

# three /// = path relative to working dir
# four //// = absolute path
db_url = 'sqlite:///data/ml_model_performance.db'
engine = create_engine(db_url)


class Base(DeclarativeBase):
    ...


# class MLModel(Base):
#     """
#     An machine learning model, e.g., "GPT4.0"
#     """
#     __tablename__ = 'ml_model'
#
#     id: Mapped[int] = mapped_column(Integer, primary_key=True)
#     #    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)  # Name of the model (e.g., ResNet, Transformer)
#     name: Mapped[str] = mapped_column(String, nullable=False)  # Name of the model (e.g., ResNet, Transformer)
#     description: Mapped[str] = mapped_column(String)  # Architecture type (e.g., "CNN", "DNN")
#     hyperparameters: Mapped[list[Hyperparameters]] = relationship("Hyperparameters", back_populates="model")
#     # Relationship: Each model can have many test results
#     #    test_results = relationship("TestResult", back_populates="model")
#     test_results: Mapped[list[ModelTestRun]] = relationship("ModelTestRun", back_populates="model")


class TestDataset:

    def __init__(self, source: str, name: str, x_train=None, y_train=None, x_test=None, y_test=None):
        self.source = source
        self.name = name
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def __repr__(self):
        return (f"TestDataset(name={self.source}.{self.name}, "
                f"x_train shape={self.x_train.shape if self.x_train is not None else None}, "
                f"y_train shape={self.y_train.shape if self.y_train is not None else None}, "
                f"x_test shape={self.x_test.shape if self.x_test is not None else None}, "
                f"y_test shape={self.y_test.shape if self.y_test is not None else None})")


def from_keras_dataset(dataset_name: str) -> TestDataset:
    dataset = getattr(datasets, dataset_name)
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    return TestDataset(source="keras", name=dataset_name,
                       x_train=x_train,
                       y_train=y_train,
                       x_test=x_test,
                       y_test=y_test
                       )

from typing import Any, Optional


class FitArgs:
    def __init__(
            self,
            x: Any = None,
            y: Any = None,
            batch_size: Optional[int] = None,
            epochs: int = 1,
            verbose: int = 1,
            callbacks: Optional[list] = None,
            validation_split: float = 0.0,
            validation_data: Optional[Any] = None,
            shuffle: bool = True,
            class_weight: Optional[dict] = None,
            sample_weight: Optional[Any] = None,
            initial_epoch: int = 0,
            steps_per_epoch: Optional[int] = None,
            validation_steps: Optional[int] = None,
            validation_batch_size: Optional[int] = None,
            validation_freq: int = 1,
            max_queue_size: int = 10,
            workers: int = 1,
            use_multiprocessing: bool = False,
    ):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.callbacks = callbacks
        self.validation_split = validation_split
        self.validation_data = validation_data
        self.shuffle = shuffle
        self.class_weight = class_weight
        self.sample_weight = sample_weight
        self.initial_epoch = initial_epoch
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.validation_batch_size = validation_batch_size
        self.validation_freq = validation_freq
        self.max_queue_size = max_queue_size
        self.workers = workers
        self.use_multiprocessing = use_multiprocessing

    def to_dict(self):
        """Converts the encapsulated arguments into a dictionary for passing to model.fit()."""
        return {
            "x": self.x,
            "y": self.y,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "verbose": self.verbose,
            "callbacks": self.callbacks,
            "validation_split": self.validation_split,
            "validation_data": self.validation_data,
            "shuffle": self.shuffle,
            "class_weight": self.class_weight,
            "sample_weight": self.sample_weight,
            "initial_epoch": self.initial_epoch,
            "steps_per_epoch": self.steps_per_epoch,
            "validation_steps": self.validation_steps,
            "validation_batch_size": self.validation_batch_size,
            "validation_freq": self.validation_freq,
            "max_queue_size": self.max_queue_size,
            "workers": self.workers,
            "use_multiprocessing": self.use_multiprocessing,
        }


class ModelTestRun(Base):
    """
    A full test definition. Specifies all fields we want to track so as to later query the db:

    Some questions we'll ask the db:
    - For a given model arch and dataset, what hyperparams yield the best point of diminishing returns on
    accuracy?
    - For a given dataset, what is the best model + hyperparams?

    A ModelTest may be run multiple times,
    for instance, in order to measure
    """
    __tablename__ = 'model_test_run'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    os_uname: Mapped[str] = mapped_column()  # e.g., "darwin", "arm"

    frontend: Mapped[str] = mapped_column()  # e.g., "keras"
    backend: Mapped[str] = mapped_column()  # e.g., "tensorflow", "jax"
    backend_proc: Mapped[str] = mapped_column()  # e.g., "cpu", "gpu"

    model_name: Mapped[str] = mapped_column(String)
    model_version: Mapped[str] = mapped_column(String)
    model_layers: Mapped[list["ModelLayer"]] = relationship(
        "ModelLayer",
        order_by="ModelLayer.sequence_number",
        collection_class=ordering_list("sequence_number"),
        back_populates="model_test_run"
    )

    fit_args: Mapped[dict] = mapped_column()

    learning_rate: Mapped[Optional[float]] = mapped_column(nullable=True)
    batch_size: Mapped[Optional[int]] = mapped_column(nullable=True)

    description: Mapped[Optional[str]] = mapped_column(nullable=True)
    # setup
    dataset_name: Mapped[str] = mapped_column()  # e.g., "CIFAR-10"
    dataset_source: Mapped[str] = mapped_column()  # e.g., "keras, adhoc"

    # compiler params (get from compile_model() call)
    #    compiler_optimizer_name: Mapped[str] = mapped_column()  # e.g., "adam"
    #    compiler_loss_name: Mapped[str] = mapped_column()  # e.g., "mse"

    # fit params
    fit_epochs: Mapped[Optional[int]] = mapped_column()

    # results
    eval_accuracy: Mapped[float] = mapped_column()  # e.g., Final test accuracy
    run_start_timestamp: Mapped[datetime] = mapped_column()  # Optional timestamp column
    run_end_timestamp: Mapped[datetime] = mapped_column(nullable=True)  # Optional timestamp column
    run_time_delta_secs: Mapped[float] = mapped_column()  # Time taken to train/test

    def __init__(self,
                 dataset: TestDataset,
                 keras_model: Model,
                 model_name,
                 description,
                 fit_args: FitArgs,
                 os_uname=f"{os.uname()}",
                 frontend="Keras",
                 backend="TensorFlow",
                 backend_proc="cpu",
                 model_version="1.0.0",
                 ):
        super().__init__()
        self.description = description
        self.run_start_timestamp = datetime.now()
        self.dataset = dataset
        self.dataset_source = dataset.source
        self.dataset_name = dataset.name
        self.keras_model = keras_model  # not directly persistent
        #
        self.os_uname = os_uname
        self.frontend = frontend
        self.backend = backend
        self.backend_proc = backend_proc
        self.model_name = model_name
        self.model_version = model_version

        for keras_layer in keras_model.layers:
            config = keras_layer.get_config()
            our_layer = ModelLayer()
            our_layer.layer_type_name = keras_layer.__class__.__name__
            self.model_layers.append(our_layer)
            for key, value in config.items():
                our_layer.hyperparameters.append(
                    Hyperparameter(var_name=key, var_value=str(value))
                )

    def compile_model(self, **kwargs):
        return self.keras_model.compile(**kwargs)

    def __repr__(self):
        return (f"TestResult(id={self.id}, model_name={self.model_name!r}, "
                f"dataset_name={self.dataset_name!r}, accuracy={self.eval_accuracy!r}, "
                f"time_taken={self.run_time_delta_secs!r})")


class ModelLayer(Base):
    __tablename__ = 'model_layer'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    model_test_run_id: Mapped[int] = mapped_column(ForeignKey('model_test_run.id'), nullable=False)
    model_test_run: Mapped["ModelTestRun"] = relationship(back_populates="model_layers")

    sequence_number: Mapped[int] = mapped_column()
    layer_type_name: Mapped[str] = mapped_column()
    hyperparameters: Mapped[list["Hyperparameter"]] = relationship()


class Hyperparameter(Base):
    __tablename__ = 'hyperparameter'

    __table_args__ = (
        UniqueConstraint('model_layer_id', 'var_name', name='_model_layer_var_name_unique'),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    model_layer_id: Mapped[int] = mapped_column(ForeignKey('model_layer.id'))

    var_name: Mapped[str] = mapped_column(nullable=False)
    var_value: Mapped[str] = mapped_column(nullable=False)


class CompileOption(Base):
    __tablename__ = 'compiler_option'
    __table_args__ = (
        UniqueConstraint('test_id', 'var_name', name='_compiler_option_name_unique'),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    test_id: Mapped[int] = mapped_column(ForeignKey('model_test_run.id'))

    var_name: Mapped[str] = mapped_column(nullable=False)
    var_value: Mapped[str] = mapped_column(nullable=False)


class RunBatch:
    """
    A group of tests to run in a single batch against the same datasets.
    """

    def __init__(self):
        self.tests: List[ModelTestRun] = []

    def add_test(self, model_test: ModelTestRun) -> RunBatch:
        self.tests.append(model_test)
        return self


def run_batch(batch: RunBatch):
    batch.results = []  # Initialize results to an empty list

    for model_test in batch.tests:
        result = run_test(model_test)
        batch.results.append(result)


def run_test(test_run: ModelTestRun):
    # Substitute different layer sequences here
    print("━" * 80)
    print(f"Running Model Test: {test_run.description}")
    print("━" * 80)

    test_run.keras_model.summary()

    dataset = test_run.dataset

    test_run.keras_model.fit(
        dataset.x_train, dataset.y_train,
        epochs=test_run.fit_epochs,
        verbose=1
    )

    loss_and_accuracy_map = test_run.keras_model.evaluate(dataset.x_test, dataset.y_test, verbose=2)

    test_run.eval_accuracy = loss_and_accuracy_map[1]

    return test_run


Session = sessionmaker(bind=engine)


def new_session():
    return Session()


def create_tables():
    Base.metadata.create_all(engine)  # Create tables

# def drop_tables():
#     Base.metadata.drop_all(engine)
