import os

import umap
import sklearn.decomposition
import pandas as pd

import cryopicls


def main():
    args = cryopicls.args.projector.parser_args()

    # Load latent representations
    if args.cryodrgn:
        Z = cryopicls.data_handling.cryodrgn.load_latent_variables(args.z_file)
    elif args.cryosparc:
        cs_file, _, _ = cryopicls.data_handling.cryosparc.find_cryosparc_files(
            args.threedvar_dir)
        Z = cryopicls.data_handling.cryosparc.load_latent_variables(cs_file)

    # Initialize projector
    if args.algorithm == 'umap':
        projector = umap.UMAP(n_neighbors=args.n_neighbors,
                              n_components=args.n_components,
                              metric=args.metric,
                              min_dist=args.min_dist,
                              random_state=args.random_state)
        axis_label = 'umap'
    elif args.algorithm == 'pca':
        projector = sklearn.decomposition.PCA(n_components=args.n_components,
                                              random_state=args.random_state)
        axis_label = 'pc'

    # Projection
    Z_proj = projector.fit_transform(Z)

    # Save result
    col_names = [f'{axis_label}_{x}' for x in range(1, Z_proj.shape[1] + 1)]
    df = pd.DataFrame(data=Z_proj, columns=col_names)
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_pickle(
        os.path.join(args.output_dir,
                     f'{args.output_file_rootname}_{args.algorithm}.pkl'))


if __name__ == '__main__':
    main()
