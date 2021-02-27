'''Perform clustering of single particle images based on their latent representations generated by cryoDRGN or cryoSPARC.'''

import sys
import os
import pickle

import numpy as np
import pandas as pd

import cryopicls


def main():
    args = cryopicls.args.clustering.parse_args()

    # Load particle metadata
    if args.cryodrgn:
        # Input is cryoDRGN result
        if os.path.splitext(args.metadata)[1] == '.csg':
            md = cryopicls.data_handling.cryosparc.CryoSPARCMetaData.load(
                args.metadata)
        elif os.path.splitext(args.metadata)[1] == '.star':
            md = cryopicls.data_handling.relion.RelionMetaData.load(
                args.metadata)
        else:
            sys.exit(
                f'--metadata {args.metadata} is neither a cryoSPARC group file nor a RELION star file!'
            )
    elif args.cryosparc:
        # Input is cryoSPARC 3D variability job
        md = cryopicls.data_handling.cryosparc.CryoSPARCMetaData.load(
            args.threedvar_csg)

    # Load latent representations, Z
    if args.cryodrgn:
        Z = cryopicls.data_handling.cryodrgn.load_latent_variables(args.z_file)
    elif args.cryosparc:
        cs_file, _ = cryopicls.data_handling.cryosparc.get_metafiles_from_csg(args.threedvar_csg)
        Z = cryopicls.data_handling.cryosparc.load_latent_variables(
            cs_file, args.threedvar_num_components)

    # Initialize clustering model
    if args.algorithm == 'auto-gmm':
        model = cryopicls.clustering.autogmm.AutoGMMClustering(**vars(args))
    elif args.algorithm == 'x-means':
        model = cryopicls.clustering.xmeans.XMeansClustering(**vars(args))
    elif args.algorithm == 'k-means':
        model = cryopicls.clustering.kmeans.KMeansClustering(**vars(args))
    elif args.algorithm == 'g-means':
        model = cryopicls.clustering.gmeans.GMeansClustering(**vars(args))

    # Do clustering
    fitted_model, cluster_labels, cluster_centers = model.fit(Z)

    # Save metadatas and model
    os.makedirs(args.output_dir, exist_ok=True)
    # The best model
    with open(os.path.join(args.output_dir,
              f'{args.output_file_rootname}_model.pkl'), 'wb') as f:
        pickle.dump(fitted_model, f, protocol=4)
    # Cluster centers
    np.savetxt(
        os.path.join(args.output_dir, f'{args.output_file_rootname}_cluster_centers.txt'),
        cluster_centers)
    # Coordinates in Z nearest to the cluster centers
    label_list = np.unique(cluster_labels)
    nearest_points = []
    for i, cluster_center in enumerate(cluster_centers):
        label = label_list[i]
        Z_cluster_center = Z[np.nonzero(cluster_labels == label)[0]]
        _, nearest_point = cryopicls.utils.nearest_in_array(
            Z_cluster_center, cluster_center)
        nearest_points.append(nearest_point)
    np.savetxt(
        os.path.join(
            args.output_dir,
            f'{args.output_file_rootname}_nearest_points_to_cluster_centers.txt'
        ), nearest_points)
    # Metadata and Z of each cluster
    for label in label_list:
        idxs = np.nonzero(cluster_labels == label)[0]
        md_cluster = md.iloc(idxs)
        md_cluster.write(args.output_dir,
                         f'{args.output_file_rootname}_cluster{label:03d}')
        Z_cluster = Z[idxs]
        np.save(
            os.path.join(args.output_dir,
                         f'{args.output_file_rootname}_cluster{label:03d}_Z'),
            Z_cluster)

    # Save Z and cluster_labels as dataframe (input for cryopicls_visualizer)
    col_names = [f'dim_{x}' for x in range(1, Z.shape[1] + 1)]
    df = pd.concat([
        pd.DataFrame(data=Z, columns=col_names),
        pd.Series(data=cluster_labels, name='cluster')
    ], axis=1)
    df.to_pickle(
        os.path.join(args.output_dir,
                     f'{args.output_file_rootname}_dataframe.pkl'))


if __name__ == '__main__':
    main()