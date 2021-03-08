import os
import pickle

from model_crnn import ModelCRNN_HQ, ModelCRNN_LQ, Model_CNN
from model_resnet import ResnetModel
from ml_tools.dataset import dataset_db_path
from ml_tools.kerasmodel import KerasModel


def train_model(run_name, conf, hyper_params, weights=None, grid_search=None):
    """Trains a model with the given hyper parameters."""
    run_name = os.path.join("train", run_name)

    # a little bit of a pain, the model needs to know how many classes to classify during initialisation,
    # but we don't load the dataset till after that, so we load it here just to count the number of labels...
    datasets_filename = dataset_db_path(conf)
    with open(datasets_filename, "rb") as f:
        dsets = pickle.load(f)
    labels = dsets[0].labels
    if conf.train.model == ResnetModel.MODEL_NAME:
        model = ResnetModel(labels, conf.train)
    elif conf.train.model == ModelCRNN_HQ.MODEL_NAME:
        model = ModelCRNN_HQ(
            labels=len(labels), train_config=conf.train, training=True, **hyper_params
        )
    elif conf.train.model == Model_CNN.MODEL_NAME:
        model = Model_CNN(
            labels=len(labels), train_config=conf.train, training=True, **hyper_params
        )
    elif conf.train.model == "keras":
        model = KerasModel(train_config=conf.train, labels=conf.labels)
    else:
        model = ModelCRNN_LQ(
            labels=len(labels), train_config=conf.train, training=True, **hyper_params
        )
    #

    model.import_dataset(datasets_filename, lbl_p=conf.train.label_probabilities)
    groups = []
    groups.append((["bird"], "bird"))
    groups.append((["hedgehog"], "hedgehog"))
    groups.append((["rodent"], "rodent"))
    groups.append((["possum", "cat"], "possum"))
    groups.append((["human"], "human"))
    groups.append((["false-positive", "insect"], "false-positive"))
    model.regroup(groups)
    # display the data set summary
    print("Training on labels", model.datasets.train.labels)
    print()
    print(
        "{:<20} {:<20} {:<20} {:<20} (segments/frames/tracks/bins)".format(
            "label", "train", "validation", "test"
        )
    )
    for label in model.datasets.train.labels:
        print(
            "{:<20} {:<20} {:<20} {:<20}".format(
                label,
                "{}/{}/{}/{:.1f}".format(*model.datasets.train.get_counts(label)),
                "{}/{}/{}/{:.1f}".format(*model.datasets.validation.get_counts(label)),
                "{}/{}/{}/{:.1f}".format(*model.datasets.test.get_counts(label)),
            )
        )
    if model.datasets.train.label_mapping:
        print("Mapped labels")
        for label in model.datasets.train.label_mapping.keys():
            print(
                "{} {:<20} {:<20} {:<20} {:<20}".format(
                    label,
                    model.datasets.train.mapped_label(label),
                    "{}/{}/{}/{:.1f}".format(*model.datasets.train.get_counts(label)),
                    "{}/{}/{}/{:.1f}".format(
                        *model.datasets.validation.get_counts(label)
                    ),
                    "{}/{}/{}/{:.1f}".format(*model.datasets.test.get_counts(label)),
                )
            )
    print(weights)
    if weights:
        model.load_weights(weights, meta=False, training=True)
    if grid_search:
        print("Searching hparams")
        model.test_hparams()
        return
    print()
    print("Training started")
    print("---------------------")
    print("Hyper parameters")
    print("---------------------")
    print(model.hyperparams_string)
    print()
    print(
        "Found {0:.1f}K training examples".format(
            model.datasets.train.sample_count / 1000
        )
    )

    model.train_model(epochs=conf.train.epochs, run_name=run_name + "_" + "TEST")
    model.save()
    model.close()

    return model
