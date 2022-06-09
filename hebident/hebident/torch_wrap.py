import torch.utils.data
import torch.nn
import torch.nn.functional
import torch.optim
import numpy

device = "cpu"


def _tensor_true_false_to_count(t: torch.Tensor):
    return t.type(torch.float).sum().item()  # noqa


class SpeciesClassifierNeuralNetwork(torch.nn.Module):  # inheriting from nn.Module!

    @staticmethod
    def to_activation_function(activation_function):
        if activation_function.lower() == "relu":
            return torch.nn.ReLU()
        elif activation_function.lower() == "mish":
            return torch.nn.Mish()
        else:
            raise RuntimeError("Only support 'relu' and 'mish' activation functions.")

    def __init__(self, num_species, num_features, layer_definition):
        # calls the init function of nn.Module.  Don't get confused by syntax,
        # just always do it in an nn.Module
        super(SpeciesClassifierNeuralNetwork, self).__init__()

        # Define the parameters that you will need.  In this case, we need A and b,
        # the parameters of the affine mapping.
        # Torch defines nn.Linear(), which provides the affine map.
        # Make sure you understand why the input dimension is vocab_size
        # and the output is num_labels!
        self.flatten = torch.nn.Flatten()
        size_of_hidden_layer = max(num_features, num_species)
        if isinstance(layer_definition, str):
            layer_definition = [layer_definition]
        num_layers = len(layer_definition)
        if num_layers < 1 or num_layers > 2:
            raise RuntimeError("We only support 1 or 2 hidden layers")
        if num_layers == 1:
            self.linear_relu_stack = torch.nn.Sequential(
                torch.nn.Linear(num_features, size_of_hidden_layer),
                self.to_activation_function(layer_definition[0]),
                torch.nn.Linear(size_of_hidden_layer, num_species))
        else:
            self.linear_relu_stack = torch.nn.Sequential(
                torch.nn.Linear(num_features, size_of_hidden_layer),
                self.to_activation_function(layer_definition[0]),
                torch.nn.Linear(size_of_hidden_layer, size_of_hidden_layer),
                self.to_activation_function(layer_definition[1]),
                torch.nn.Linear(size_of_hidden_layer, num_species),
            )

        # self.linear = torch.nn.Linear(num_features, num_species)

        # NOTE! The non-linearity log softmax does not have parameters! So we don't need
        # to worry about that here

    def forward(self, features_vec):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional

        # PB changed: return torch.nn.functional.log_softmax(self.linear(bow_vec), dim=1)
        # to
        return self.linear_relu_stack(features_vec)
        # return torch.nn.functional.log_softmax(self.linear(features_vec), dim=1)


class TorchClassifier:

    def __init__(self,
                 optimizer_type,
                 amsgrad,
                 max_epochs,
                 learning_rate,
                 layer_definition,
                 batch_size=10,
                 shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.optimizer_type = optimizer_type.lower()
        self.amsgrad = amsgrad
        self.layer_definition = layer_definition
        self.model = None
        self.classes = None

    def train_one_step(self, dataloader, loss_fn, optimizer):
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction error
            pred = self.model(X.to(device))
            loss = loss_fn(pred, y.to(device))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # def _test_impl(self, dataloader):
    #    size = len(dataloader.dataset)
    #    self.model.eval()
    #    correct = 0
    #    with torch.no_grad():
    #        for X, y in dataloader:
    #            X, y = X.to(device), y.to(device)
    #            pred = self.model(X)
    #            prob_for_batch = torch.nn.Softmax(dim=1)(pred)
    #            _, indexes = torch.sort(prob_for_batch, descending=True)
    #
    #            this_correct = _tensor_true_false_to_count(pred.argmax(1) == y)  # noqa
    #            correct += this_correct
    #
    #    correct /= size
    #    return correct

    # def test(self, dataloader_test, best):
    #    test_correct = self._test_impl(dataloader_test)
    #    if not best or test_correct >= best:
    #        best = test_correct, copy.deepcopy(self.model.state_dict())
    #    return best

    @staticmethod
    def _make_index_dict_and_array(input_series):
        next_index = 0
        values_dict = {}  # The mapping from name to index
        values_array = []  # The mapping from index to name

        for row in input_series:
            if row not in values_dict:
                values_dict[row] = next_index
                values_array.append(row)
                next_index += 1
        return values_dict, values_array

    @staticmethod
    def _make_target_tensor(input_series, index_dict):
        # https://sparrow.dev/pytorch-one-hot-encoding/
        # https://discuss.pytorch.org/t/runtimeerror-multi-target-not-supported-newbie/10216/2
        #   ^- Because we are using CrossEntropyLoss we need to pass class indexes, NOT one-hot encoded array
        values_array = []
        for row in input_series:
            if row not in index_dict:
                raise RuntimeError(f"'{row}' is not in index_dict, which has {len(index_dict)} entries")
            values_array.append(index_dict[row])
        return torch.tensor(values_array)

    def fit(self, training_df, training_target, testing_df, testing_target, print_function):
        # print(f"{type(training_df)} :: {training_df}")
        # print(f"{type(training_target)} :: {training_target}")
        # print(list(training_df.columns))
        # training_df.to_csv(r"C:\Users\pete\AppData\Local\Temp\training_df.csv", na_rep="NA")

        index_dict, self.classes = self._make_index_dict_and_array(training_target)
        num_species = len(index_dict)
        num_features = len(training_df.columns)  # In this formulation every column in the training data is a feature to use

        training_features = torch.from_numpy(numpy.array(training_df, dtype=numpy.float32))
        training_target_tensor = self._make_target_tensor(training_target, index_dict)
        training_dataset = torch.utils.data.TensorDataset(training_features, training_target_tensor)
        train_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        # testing_features = torch.from_numpy(numpy.array(testing_df, dtype=numpy.float32))
        # testing_target_tensor = self._make_target_tensor(testing_target, index_dict)
        # testing_dataset = torch.utils.data.TensorDataset(testing_features, testing_target_tensor)
        # testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        self.model = SpeciesClassifierNeuralNetwork(num_species, num_features, self.layer_definition)

        # print(f"{index_dict}")
        # print("Model created; will now seek to optimize")

        # no_weights = None
        # class_weights = no_weights
        loss_function = torch.nn.CrossEntropyLoss()
        if self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, amsgrad=self.amsgrad)
        elif self.optimizer_type == "adamw":
            # params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, amsgrad=self.amsgrad)
        else:
            raise RuntimeError(f"{self.optimizer_type} unknown type (try sgd, adam, adamw)")
        # best = None
        print_function("Fitting", end="")
        for t in range(self.max_epochs):
            print_function(".", end="")
            self.train_one_step(train_dataloader, loss_function, optimizer)
            # best = self.test(train_dataloader, best)
        print_function("")

        # At the end, restore the best state dict
        # self.model.load_state_dict(best[1])

    def predict(self, regress_values):
        # if not self.model_state_injected:
        #    raise RuntimeError("Can't predict - model hasn't had state injected")
        prev = self.model.training
        self.model.train(False)
        mapped = self.model(torch.tensor([regress_values]).float().to(device))
        res = torch.nn.Softmax(dim=1)(mapped)[0].tolist()
        self.model.train(prev)
        return res

    def class_names(self):
        return self.classes
