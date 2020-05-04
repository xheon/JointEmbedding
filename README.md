# Joint Embedding of 3D Scan and CAD Objects [ICCV '19] 
![Joint Embedding of 3D Scan and CAD Objects](/images/teaser.jpg)

We learn a joint embedding between cluttered, incomplete 3D scanned objects and CAD models, where semantically similar objects of both domains lie close together regardless of the presence of low-level geometric differences. Such an embedding space enables for instance efficient nearest-neighbor cross-domain retrieval.

For the task of CAD model retrieval other approaches often evaluate the performance on a class category level, i.e., comparing the class category of the query to those of the results. However, we wish to evaluate the retrieval performance on a more fine-grained level than class categories. 
To this end, we propose our new **Scan-CAD Object Similarity** dataset.

For more information, please have a look at our [ICCV 2019 paper](https://arxiv.org/abs/1908.06989 "ICCV 2019 Paper").


## Scan-CAD Object Similarity Dataset
We present the *Scan-CAD Object Similarity* dataset consisting of ```5102``` ranked scan-CAD similarity annotations.

To construct this dataset we use [ScanNet](http://www.scan-net.org/ "ScanNet"), [Scan2CAD](https://github.com/skanti/Scan2CAD "Scan2CAD") and [ShapeNetCore.v2](https://shapenet.org/ "ShapeNetCore.v2").
We extract the regions of scanned objects from ScanNet scenes using its semantic segmentations and bring the object into a canonical pose using the correspondences from Scan2CAD.

To build a ranked list of similar CAD models, we developed an annotation interface where the annotator sees 6 proposed CAD models, which are similar to a specific scan query, selects up to 3 most similar CAD models from this pool and ranks them.

If you would like to get access to the Scan-CAD Object Similarity dataset, please fill out this [google form](https://forms.gle/3BhUaU1JaECWrbTw6 "google form to get access"), and once accepted, we will send out the download link.

## Video
You can view our YouTube video [here](https://www.youtube.com/watch?v=-RxSlQ6tOEA).

[![Our YouTube Video](https://img.youtube.com/vi/-RxSlQ6tOEA/0.jpg)](https://www.youtube.com/watch?v=-RxSlQ6tOEA)


## Citation
If you use the data or code please cite:

```
@inproceedings{dahnert2019embedding,
    title={Joint Embedding of 3D Scan and CAD Objects},
    author={Dahnert, Manuel and Dai, Angela and Guibas, Leonidas and Nie{\ss}ner, Matthias},
	booktitle={The IEEE International Conference on Computer Vision (ICCV)},
	year={2019}
}
```

## Contact
If you have any questions, please contact us at [scan-cad-similarity@googlegroups.com](mailto:scan-cad-similarity@googlegroups.com).