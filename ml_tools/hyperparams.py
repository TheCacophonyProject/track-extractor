from ml_tools.dataset import TrackChannels


class HyperParams(dict):
    """ Defines hyper paramaters for model """

    def __init__(self, *args):
        super(HyperParams, self).__init__(*args)

        filtered = self.get("use_filtered")
        if filtered:
            self["channel"] = TrackChannels.filtered

        self.insert_defaults()

    def insert_defaults(self):
        self["model"] = self.model
        self["dense_sizes"] = self.dense_sizes
        self["base_training"] = self.base_training
        self["retrain_layer"] = self.retrain_layer
        self["dropout"] = self.dropout
        self["learning_rate"] = self.learning_rate
        self["learning_rate_decay"] = self.learning_rate_decay
        self["use_movement"] = self.use_movement
        self["model"] = self.model
        self["use_segments"] = self.use_segments
        self["square_width"] = self.square_width
        self["buffer_size"] = self.buffer_size
        self["frame_size"] = self.frame_size

        self["shuffle"] = self.shuffle
        self["train_load_threads"] = self.train_load_threads
        self["channel"] = self.channel

    @property
    def output_dim(self):
        output_dim = (self.frame_size, self.frame_size, 3)
        if self.use_movement:
            output_dim = (
                output_dim[0] * self.square_width,
                output_dim[1] * self.square_width,
                3,
            )
        return output_dim

    # Model hyper paramters
    @property
    def model(self):
        return self.get("model", "resnetv2")

    @property
    def channel(self):
        return self.get("channel", TrackChannels.thermal)

    @property
    def dense_sizes(self):
        return self.get("dense_sizes", [1024, 512])

    @property
    def label_smoothing(self):
        return self.get("label_smoothing", 0)

    @property
    def base_training(self):
        return self.get("base_training", False)

    @property
    def retrain_layer(self):
        return self.get("retrain_layer")

    @property
    def dropout(self):
        return self.get("dropout")

    @property
    def learning_rate(self):
        return self.get("learning_rate")

    @property
    def learning_rate_decay(self):
        return self.get("learning_rate_decay", 1.0)

    # Datageneration parameters
    @property
    def batch_size(self):
        return self.get("batch_size", 32)

    @property
    def lstm(self):
        return self.get("lstm", False)

    @property
    def use_movement(self):
        return self.get("use_movement", True)

    @property
    def use_segments(self):
        return self.get("use_segments", True)

    @property
    def square_width(self):
        return self.get("square_width", 5)

    @property
    def buffer_size(self):
        return self.get("buffer_size", 128)

    @property
    def frame_size(self):
        return self.get("frame_size", 32)

    @property
    def shuffle(self):
        return self.get("shuffle", True)

    @property
    def train_load_threads(self):
        return self.get("train_load_threads", 2)
