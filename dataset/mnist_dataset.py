import codecs
import os
import sys
import pathlib
import tempfile
import substra

import numpy as np
from torchvision.datasets import MNIST


def get_int(b: bytes) -> int:
    return int(codecs.encode(b, "hex"), 16)


def MNISTraw2numpy(path: str, strict: bool = True) -> np.array:
    # read
    with open(path, "rb") as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    assert 1 <= nd <= 3
    numpy_type = np.uint8
    s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]

    num_bytes_per_value = np.iinfo(numpy_type).bits // 8
    # The MNIST format uses the big endian byte order. If the system uses little endian byte order by default,
    # we need to reverse the bytes before we can read them with np.frombuffer().
    needs_byte_reversal = sys.byteorder == "little" and num_bytes_per_value > 1
    parsed = np.frombuffer(bytearray(data), dtype=numpy_type, offset=(4 * (nd + 1)))
    if needs_byte_reversal:
        parsed = parsed.flip(0)

    assert parsed.shape[0] == np.prod(s) or not strict
    return parsed.reshape(*s)


def setup_mnist(data_path, output_path, data_organizations):
    raw_path = pathlib.Path(data_path) / "MNIST" / "raw"
    n_clients = len(data_organizations)

    # Download the dataset
    MNIST(data_path, download=True)

    # Extract numpy arrays from raw data
    train_images = MNISTraw2numpy(str(raw_path / "train-images-idx3-ubyte"))
    train_labels = MNISTraw2numpy(str(raw_path / "train-labels-idx1-ubyte"))
    test_images = MNISTraw2numpy(str(raw_path / "t10k-images-idx3-ubyte"))
    test_labels = MNISTraw2numpy(str(raw_path / "t10k-labels-idx1-ubyte"))

    # Split arrays into the number of organizations
    # For the training data, we split the data according to the label,
    # to mimic different data distribution between organizations
    # For testing, data is split uniformly between organizations
    masks = [train_labels%n_clients == i for i in range(n_clients)]
    train_images_folds = [train_images[masks[i]] for i in range(n_clients)]
    train_labels_folds = [train_labels[masks[i]] for i in range(n_clients)]
    test_images_folds = np.split(test_images, n_clients)
    test_labels_folds = np.split(test_labels, n_clients)

    # Save splits in different folders to simulate the different organizations
    client_data = zip(data_organizations, train_images_folds, train_labels_folds, test_images_folds, test_labels_folds)
    for client, train_data, train_label, test_data, test_label in client_data :
        org_id = client.organization_info().organization_id

        # Save train dataset on each org
        os.makedirs(str(output_path / org_id / "train"), exist_ok=True)
        np.save(str(output_path / org_id / "train/train_images.npy"), train_data)
        np.save(str(output_path / org_id / "train/train_labels.npy"), train_label)

        # Save test dataset on each org
        os.makedirs(str(output_path / org_id / "test"), exist_ok=True)
        np.save(str(output_path / org_id / "test/test_images.npy"), test_data)
        np.save(str(output_path / org_id / "test/test_labels.npy"), test_label)
        

def data_registration(data_organizations):
    dataset_keys = {}
    train_datasample_keys = {}
    test_datasample_keys = {}

    with tempfile.TemporaryDirectory() as tmp_dir: 
        data_path = pathlib.Path(tmp_dir) / "data_mnist"
        output_path = pathlib.Path.cwd() / "organizations_data"
        setup_mnist(data_path, output_path, data_organizations)
        

    assets_directory = pathlib.Path.cwd() / "dataset"

    for client in data_organizations:
        org_id = client.organization_info().organization_id

        # We set up all permissions to public for simplicity in this tutorial
        permissions_dataset = substra.sdk.schemas.Permissions(public=True, authorized_ids=[org_id])

        # DatasetSpec is the specification of a dataset. It makes sure every field
        # is well-defined, and that our dataset is ready to be registered.
        # The real dataset object is created in the add_dataset method.

        dataset = substra.sdk.schemas.DatasetSpec(
            name="MNIST",
            type="npy",
            data_opener=assets_directory / "mnist_opener.py",
            description=assets_directory / "description.md",
            permissions=permissions_dataset,
            logs_permission=permissions_dataset,
        )
        dataset_keys[org_id] = client.add_dataset(dataset)

        # Add the training data on each organization.
        data_sample = substra.sdk.schemas.DataSampleSpec(
            data_manager_keys=[dataset_keys[org_id]],
            path=output_path / org_id / "train",
        )
        train_datasample_keys[org_id] = client.add_data_sample(data_sample)

        # Add the testing data on each organization.
        data_sample = substra.sdk.schemas.DataSampleSpec(
            data_manager_keys=[dataset_keys[org_id]],
            path=output_path / org_id / "test",
        )
        test_datasample_keys[org_id] = client.add_data_sample(data_sample)
    return dataset_keys, train_datasample_keys, test_datasample_keys

