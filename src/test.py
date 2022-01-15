import argparse
import json
from typing import List, Tuple, Dict

import numpy as np
import scipy.spatial
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import data
import metrics
import utils
from models import *


def main(opt: argparse.Namespace):
    # Configure environment
    utils.set_gpu(opt.gpu)
    device = torch.device("cuda")

    # Configure networks
    separation_model: nn.Module = SeparationNet(ResNetEncoder(1, [16, 32, 64, 128, 512]),
                                                ResNetDecoder(1),
                                                ResNetDecoder(1))
    completion_model: nn.Module = HourGlass(ResNetEncoder(1),
                                            ResNetDecoder(1))

    triplet_model: nn.Module = TripletNet(ResNetEncoder(1))

    # Load checkpoints
    separation_model.load_state_dict(torch.load(opt.model_path + "_separation.pt"))
    completion_model.load_state_dict(torch.load(opt.model_path + "_completion.pt"))
    triplet_model.load_state_dict(torch.load(opt.model_path + "_triplet.pt"))

    separation_model = separation_model.to(device)
    completion_model = completion_model.to(device)
    triplet_model = triplet_model.to(device)

    # Make sure models are in evaluation mode
    separation_model.eval()
    completion_model.eval()
    triplet_model.eval()

    # Compute domain confusion
    train_confusion_results = evaluate_confusion(separation_model, completion_model, triplet_model,
                                                 device, opt.confusion_train_path, opt.scannet_path, opt.shapenet_path,
                                                 opt.confusion_num_neighbors)
    print(train_confusion_results)

    val_confusion_results = evaluate_confusion(separation_model, completion_model, triplet_model,
                                               device, opt.confusion_val_path, opt.scannet_path, opt.shapenet_path,
                                               opt.confusion_num_neighbors)
    print(val_confusion_results)

    # Compute similarity metrics
    train_retrieval_accuracy = evaluate_similarity_metrics(separation_model, completion_model, triplet_model,
                                                           device, opt.similarity_path,
                                                           opt.scannet_path, opt.shapenet_path)
    # setup tsne
    # todo



def evaluate_confusion(separation: nn.Module, completion: nn.Module, triplet: nn.Module, device, dataset_path: str,
                       scannet_path: str, shapenet_path: str, num_neighbors: int) -> Tuple[np.array, list]:
    # Configure datasets
    dataset: Dataset = data.Scan2Cad(dataset_path, scannet_path, shapenet_path, "all", ["validation"], scan_rep="sdf",
                                     transformation=data.to_occupancy_grid)
    batch_size = 1
    dataloader: DataLoader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=0)

    embeddings: List[torch.Tensor] = []  # contains all embedding vectors
    names: List[str] = []  # contains the names of the samples
    domains: List[int] = []  # contains number labels for domains (scan=0/cad=1)

    # Iterate over data
    for scan, cad in tqdm(dataloader, total=len(dataloader)):
        # Move data to GPU
        scan_data = scan["content"].to(device)
        cad_data = cad["content"].to(device)

        with torch.no_grad():
            # Pass scan through networks
            scan_foreground, _ = separation(torch.sigmoid(scan_data))
            scan_completed = completion(torch.sigmoid(scan_foreground))
            scan_latent = triplet.embed(torch.sigmoid(scan_completed)).view(batch_size, -1)
            embeddings.append(scan_latent)
            names.append(f"/scan/{scan['name']}")
            domains.append(0)

            # Embed cad
            cad_latent = triplet.embed(cad_data).view(batch_size, -1)
            embeddings.append(cad_latent)
            names.append(f"/cad/{cad['name']}")
            domains.append(1)

    embedding_space = torch.cat(embeddings, dim=0)
    embedding_space = embedding_space.cpu().numpy()

    domain_labels: np.array = np.asarray(domains)

    # Compute distances between all samples
    distance_matrix = metrics.compute_distance_matrix(embedding_space)
    confusion, conditional_confusions = metrics.compute_knn_confusions(distance_matrix, domain_labels, num_neighbors)
    confusion_mean = np.average(confusion)
    conditional_confusions_mean = [np.average(conf) for conf in conditional_confusions]

    return confusion_mean, conditional_confusions_mean


def evaluate_similarity_metrics(separation: nn.Module, completion: nn.Module, triplet: nn.Module, device,
                                dataset_path: str, scannet_path: str, shapenet_path: str) -> None:
    unique_scan_objects, unique_cad_objects = get_unique_samples(dataset_path)

    batch_size = 1
    scan_dataset: Dataset = data.FileListDataset(scannet_path, unique_scan_objects, ".sdf",
                                                 transformation=data.to_occupancy_grid)
    scan_dataloader = torch.utils.data.DataLoader(dataset=scan_dataset, shuffle=False, batch_size=batch_size)

    # Evaluate all unique scan embeddings
    embeddings: Dict[str, np.array] = {}
    for name, element in tqdm(scan_dataloader, total=len(scan_dataloader)):
        # Move data to GPU
        element = element.to(device)
        with torch.no_grad():
            scan_foreground, _ = separation(torch.sigmoid(element))
            scan_completed = completion(torch.sigmoid(scan_foreground))
            scan_latent = triplet.embed(torch.sigmoid(scan_completed)).view(-1)

        embeddings[name[0]] = scan_latent.cpu().numpy().squeeze()

    # Evaluate all unique cad embeddings
    cad_dataset: Dataset = data.FileListDataset(shapenet_path, unique_cad_objects, "__0__.df",
                                                transformation=data.to_occupancy_grid)
    cad_dataloader = torch.utils.data.DataLoader(dataset=cad_dataset, shuffle=False, batch_size=batch_size)

    for name, element in tqdm(cad_dataloader, total=len(cad_dataloader)):
        # Move data to GPU
        element = element.to(device)
        with torch.no_grad():
            cad_latent = triplet.embed(element).view(-1)

        embeddings[name[0]] = cad_latent.cpu().numpy().squeeze()

    # embedding_vectors = np.load("/mnt/raid/dahnert/joint_embedding_binary/embedding_vectors.npy")
    # embedding_names = json.load(open("/mnt/raid/dahnert/joint_embedding_binary/embedding_names.json"))
    # embeddings = dict(zip(embedding_names, embedding_vectors))

    # Evaluate metrics
    with open(dataset_path) as f:
        samples = json.load(f).get("samples")

        retrieved_correct = 0
        retrieved_total = 0

        ranked_correct = 0
        ranked_total = 0

        selected_categories = ["02747177", "02808440", "02818832", "02871439", "02933112", "03001627", "03211117", "03337140", "04256520", "04379243", "other"]
        per_category_retrieved_correct = {category: 0 for category in selected_categories}
        per_category_retrieved_total = {category: 0 for category in selected_categories}

        per_category_ranked_correct = {category: 0 for category in selected_categories}
        per_category_ranked_total = {category: 0 for category in selected_categories}

        # Iterate over all annotations
        for sample in tqdm(samples, total=len(samples)):
            reference_name = sample["reference"]["name"].replace("/scan/", "")
            reference_embedding = embeddings[reference_name][np.newaxis, :]

            pool_names = np.asarray([p["name"].replace("/cad/", "") for p in sample["pool"]])
            pool_embeddings = [embeddings[p] for p in pool_names]
            pool_embeddings = np.asarray(pool_embeddings)

            # Compute distances in embedding space
            distances = scipy.spatial.distance.cdist(reference_embedding, pool_embeddings, metric="euclidean")
            sorted_indices = np.argsort(distances, axis=1)
            sorted_distances = np.take_along_axis(distances, sorted_indices, axis=1)
            sorted_distances = sorted_distances[0]

            predicted_ranking = np.take(pool_names, sorted_indices)[0].tolist()

            ground_truth_names = [r["name"].replace("/cad/", "") for r in sample["ranked"]]

            # retrieval accuracy
            sample_retrieved_correct = 1 if metrics.is_correctly_retrieved(predicted_ranking, ground_truth_names) else 0
            retrieved_correct += sample_retrieved_correct
            retrieved_total += 1

            # per-category retrieval accuracy
            reference_category = metrics.get_category_from_list(metrics.get_category(reference_name), selected_categories)
            per_category_retrieved_correct[reference_category] += sample_retrieved_correct
            per_category_retrieved_total[reference_category] += 1

            # ranking quality
            sample_ranked_correct = metrics.count_correctly_ranked_predictions(predicted_ranking, ground_truth_names)
            ranked_correct += sample_ranked_correct
            ranked_total += len(ground_truth_names)

            per_category_ranked_correct[reference_category] += sample_ranked_correct
            per_category_ranked_total[reference_category] += len(ground_truth_names)

        print(f"correct: {retrieved_correct}, total: {retrieved_total}, accuracy: {retrieved_correct/retrieved_total}")

        for (category, correct), total in zip(per_category_retrieved_correct.items(), per_category_retrieved_total.values()):
            print(f"{category}: {correct:>5d}/{total:>5d} --> {correct/total:4.3f}")

        print(f"correct: {ranked_correct}, total: {ranked_total}, accuracy: {ranked_correct/ranked_total}")

        for (category, correct), total in zip(per_category_ranked_correct.items(), per_category_ranked_total.values()):
            print(f"{category}: {correct:>5d}/{total:>5d} --> {correct/total:4.3f}")

    return None


def get_unique_samples(dataset_path: str) -> Tuple[List[str], List[str]]:
    unique_scans = set()
    unique_cads = set()

    with open(dataset_path) as f:
        samples = json.load(f).get("samples")

    for sample in tqdm(samples, desc="Gather unique samples"):
        scan = sample["reference"]["name"].replace("/scan/", "")
        unique_scans.add(scan)

        for cad_element in sample["pool"]:
            cad = cad_element["name"].replace("/cad/", "")
            unique_cads.add(cad)

    print(f"Unique scan samples: {len(unique_scans)}, unique cad element {len(unique_cads)}")

    return list(unique_scans), list(unique_cads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Test script for Joint Embedding (ICCV 2019)")
    parser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    parser.add_argument("--scannet_path", type=str)
    parser.add_argument("--shapenet_path", type=str)

    parser.add_argument("--confusion_train_path", type=str, help="Path to the json file containing a 1:1 mapping.")
    parser.add_argument("--confusion_val_path", type=str, help="Path to the json file containing a 1:1 mapping.")
    parser.add_argument("--confusion_num_neighbors", type=int, default=10)

    parser.add_argument("--similarity_path", type=str)

    parser.add_argument("--model_path", type=str,
                        help="Path to the model checkpoint file, excluding _triplet.pt. E.g. /path/2019-01-01_01:00:00_200000")
    args = parser.parse_args()
    main(args)
