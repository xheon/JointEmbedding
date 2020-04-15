import argparse
import json
import os
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import data
import utils
from models import *


def forward(scan, cad, negative, separation_model, completion_model, triplet_model,
            criterion_separation, criterion_completion, criterion_triplet, device):
    # Prepare scan sample
    scan_model, scan_mask, scan_name = scan["content"], scan["mask"], scan["name"]
    scan_bg_mask = torch.where(scan_mask == 0, scan_model, torch.zeros(scan_mask.shape))
    scan_model = scan_model.to(device, non_blocking=True)
    scan_fg_mask = scan_mask.to(device, non_blocking=True)
    scan_bg_mask = scan_bg_mask.to(device, non_blocking=True)

    # Prepare CAD sample
    cad_model = cad["content"]
    cad_model = cad_model.to(device, non_blocking=True)

    # Prepare negative sample
    negative_model = negative["content"]
    negative_model = negative_model.to(device, non_blocking=True)

    # Pass data through networks
    # 1) Separate foreground and background
    foreground, background = separation_model(torch.sigmoid(scan_model))
    loss_foreground = torch.mean(criterion_separation(foreground, scan_fg_mask), dim=[1, 2, 3, 4]).mean()
    loss_background = torch.mean(criterion_separation(background, scan_bg_mask), dim=[1, 2, 3, 4]).mean()

    # 2) Complete foreground w.r.t. CAD model
    completed = completion_model(torch.sigmoid(foreground))
    loss_completion = torch.mean(criterion_completion(completed, cad_model), dim=[1, 2, 3, 4]).mean()

    # 3) Embed completed output as a triplet
    # anchor: completed output, positive: CAD model, negative: random CAD sample
    anchor, positive, negative = triplet_model(torch.sigmoid(completed), cad_model, negative_model)
    a, p, n = anchor.view(anchor.shape[0], -1), positive.view(anchor.shape[0], -1), negative.view(
        anchor.shape[0], -1)
    loss_triplet = criterion_triplet(a, p, n).mean()

    return loss_foreground, loss_background, loss_completion, loss_triplet


def main(opt: argparse.Namespace) -> None:
    utils.set_gpu(opt.gpu)
    device = torch.device("cuda")
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_path = os.path.join(opt.output_root, run_name)
    print(f"Start training {run_path}")
    print(vars(opt))

    # Save config
    os.makedirs(run_path, exist_ok=True)
    with open(os.path.join(run_path, "config.json"), "w") as f:
        json.dump(vars(opt), f, indent=4)

    # Data
    train_dataset: Dataset = data.Scan2Cad(opt.scan2cad_file, opt.scannet_path, opt.shapenet_path, "train", ["train"],
                                           rotation=opt.rotation_augmentation, flip=opt.flip_augmentation,
                                           jitter=opt.jitter_augmentation, transformation=data.to_occupancy_grid,
                                           scan_rep="sdf", load_mask=True, add_negatives=True)
    train_dataloader: DataLoader = DataLoader(train_dataset, shuffle=True, batch_size=opt.batch_size,
                                              num_workers=opt.num_workers, pin_memory=True)

    val_dataset: Dataset = data.Scan2Cad(opt.scan2cad_file, opt.scannet_path, opt.shapenet_path, "validation",
                                         ["validation"], rotation=opt.rotation_augmentation, flip=opt.flip_augmentation,
                                         jitter=opt.jitter_augmentation, transformation=data.to_occupancy_grid,
                                         scan_rep="sdf", load_mask=True, add_negatives=True)
    val_dataloader: DataLoader = DataLoader(val_dataset, shuffle=False, batch_size=opt.batch_size,
                                            num_workers=opt.num_workers, pin_memory=True)

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

            losses = forward(scan, cad, negative, separation_model, completion_model, triplet_model,
                             criterion_separation, criterion_completion, criterion_triplet, device)

            loss_foreground, loss_background, loss_completion, loss_triplet = losses
            loss_total = loss_foreground + loss_background + loss_completion + loss_triplet

            # Train step
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # Log to console
            if iteration_number % opt.log_frequency == opt.log_frequency - 1:
                print(f"[E{epoch:04d}, I{iteration_number:05d}]\tTotal: {loss_total: 05.3f}",
                      f"\tFG: {loss_foreground: 05.3f}\tBG: {loss_background: 05.3f}",
                      f"\tCompletion: {loss_completion: 05.3f} \tTriplet: {loss_triplet: 05.3f}")

            # Validate
            if iteration_number % opt.validate_frequency == opt.validate_frequency - 1:
                with torch.no_grad():
                    separation_model.eval()
                    completion_model.eval()
                    triplet_model.eval()

                    val_losses = defaultdict(list)

                    # Go through entire validation set
                    for _, (scan_v, cad_v, negative_v) in tqdm(enumerate(val_dataloader),
                                                               total=len(val_dataloader), leave=False):
                        losses = forward(scan_v, cad_v, negative_v, separation_model, completion_model, triplet_model,
                                         criterion_separation, criterion_completion, criterion_triplet, device)

                        loss_foreground, loss_background, loss_completion, loss_triplet = losses
                        loss_total = loss_foreground + loss_background + loss_completion + loss_triplet
                        val_losses["FG"].append(loss_foreground.item())
                        val_losses["BG"].append(loss_background.item())
                        val_losses["Completion"].append(loss_completion.item())
                        val_losses["Triplet"].append(loss_triplet.item())
                        val_losses["Total"].append(loss_total.item())

                    # Aggregate losses
                    val_losses_summary = {k: torch.mean(torch.tensor(v)) for k, v in val_losses.items()}
                    print(f"-Val [E{epoch:04d}, I{iteration_number:05d}]\tTotal: {val_losses_summary['Total']:05.3f}",
                          f"\tFG: {val_losses_summary['FG']:05.3f} \tBG: {val_losses_summary['BG']:05.3f}",
                          f"\tCompletion: {val_losses_summary['Completion']:05.3f}",
                          f"\tTriplet: {val_losses_summary['Triplet']:05.3f}")

            # Save checkpoint
            if iteration_number % opt.checkpoint_frequency == opt.checkpoint_frequency - 1:
                checkpoint_name = f"{run_name}_{iteration_number:05d}"

                torch.save(separation_model.state_dict(),
                           os.path.join(run_path, f"{checkpoint_name}_separation.pt"))
                torch.save(completion_model.state_dict(),
                           os.path.join(run_path, f"{checkpoint_name}_completion.pt"))
                torch.save(triplet_model.state_dict(), os.path.join(run_path, f"{checkpoint_name}_triplet.pt"))
                print(f"Saved model at {run_path}/{checkpoint_name}")

            iteration_number += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Joint Embedding (ICCV 2019)")
    parser.add_argument("--gpu", type=int, default=0, help="which gpu to use")
    parser.add_argument("--batch_size", type=int, default=128, help="how many samples per batch")
    parser.add_argument("--num_epochs", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--scannet_path", type=str)
    parser.add_argument("--shapenet_path", type=str)
    parser.add_argument("--scan2cad_file", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--triplet_margin", type=float, default=1e-2)
    parser.add_argument("--rotation_augmentation", type=str, default="interpolation", help="fixed, interpolation, none")
    parser.add_argument("--flip_augmentation", type=bool, default=False)
    parser.add_argument("--jitter_augmentation", type=bool, default=False)
    parser.add_argument("--log_frequency", type=int, default=10)
    parser.add_argument("--validate_frequency", type=int, default=250)
    parser.add_argument("--checkpoint_frequency", type=int, default=1000)
    parser.add_argument("--output_root", type=str)
    args = parser.parse_args()
    main(args)
