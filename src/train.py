import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import utils
from models import *


def main(opt: argparse.Namespace) -> None:
    utils.set_gpu(opt.gpu)
    device = torch.device("cuda")

    # Data
    train_dataset: Dataset = None
    train_dataloader: DataLoader = None

    val_dataset: Dataset = None
    val_dataloader: DataLoader = None

    # Models
    separation_model: nn.Module = SeparationNet(ResNetEncoder(1, [16, 32, 64, 128, 512]),
                                                ResNetDecoder(1),
                                                ResNetDecoder(1))
    completion_model: nn.Module = HourGlass(ResNetEncoder(1),
                                            ResNetDecoder(1))

    triplet_model: nn.Module = TripletNet(ResNetEncoder(1))

    separation_model = separation_model.to(device)
    completion_model = completion_model.to(device)
    triplet_model = triplet_model.to(device)

    model_parameters = list(separation_model.parameters()) + \
                       list(completion_model.parameters()) + \
                       list(triplet_model.parameters())

    optimizer = optim.Adam(model_parameters, lr=opt.learning_rate, weight_decay=opt.weight_decay)

    criterion_separation = nn.BCEWithLogitsLoss(reduction="none")
    criterion_completion = nn.BCEWithLogitsLoss(reduction="none")
    criterion_triplet = nn.TripletMarginLoss(reduction="none", margin=opt.triplet_margin)

    # Main loop
    iteration_number = 0

    for epoch in range(opt.num_epochs):
        train_dataloader.dataset.regenerate_negatives()

        for _, (scan, cad, negative) in enumerate(train_dataloader):
            utils.stepwise_learning_rate_decay(optimizer, opt.learning_rate, iteration_number, [40000, 80000, 120000])

            separation_model.train()
            completion_model.train()
            triplet_model.train()

            # Prepare scan sample
            scan_model, scan_mask, scan_name = scan["content"], scan["mask"], scan["name"]
            scan_bg_mask = torch.where(scan_mask == 0, scan_model, torch.zeors(scan_mask.shape))
            scan_model = scan_model.to(device, non_blocking=True)
            scan_fg_mask = scan_mask.to(device, non_blocking=True)
            scan_bg_mask = scan_bg_mask.to(device, non_blocking=True)

            # Prepare CAD sample
            cad_model = cad["model"]
            cad_model = cad_model.to(device, non_blocking=True)

            # Prepare negative sample
            negative_model = negative["content"]
            negative_model = negative_model.to(device, non_blocking=True)

            # Pass data through networks
            # 1) Separate foreground and background
            foreground, background = separation_model(scan_model)
            loss_foreground = torch.mean(criterion_separation(foreground, scan_fg_mask), dim=[1, 2, 3, 4]).mean()
            loss_background = torch.mean(criterion_separation(background, scan_bg_mask), dim=[1, 2, 3, 4]).mean()

            # 2) Complete foreground w.r.t. CAD model
            completed = completion_model(foreground)
            loss_completion = torch.mean(criterion_completion(completed, cad_model), dim=[1, 2, 3, 4]).mean()

            # 3) Embed completed output as a triplet
            # anchor: completed output, positive: CAD model, negative: random CAD sample
            anchor, positive, negative = triplet_model(completed, cad_model, negative_model)
            a, p, n = anchor.view(anchor.shape[0], -1), positive.view(anchor.shape[0], -1), negative.view(anchor.shape[0], -1)
            loss_triplet = criterion_triplet(a, p, n).mean()

            # Train step
            optimizer.zero_grad()
            loss_total = loss_foreground + loss_background + loss_completion + loss_triplet
            loss_total.backward()
            optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dahnert et al. - Joint Embedding (ICCV 2019)")
    parser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    parser.add_argument("--batch_size", type=int, default=128, help="how many samples per batch")
    parser.add_argument("--num_epochs", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--scannet_root_path", type=str)
    parser.add_argument("--shapenet_root_path", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--triplet_margin", type=float, default=1e-2)
    args = parser.parse_args()
    main(args)
