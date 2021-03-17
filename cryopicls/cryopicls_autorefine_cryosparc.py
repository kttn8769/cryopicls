'''Perform auto-refinement of clusters found by cryoPICLS'''

import glob

import cryopicls


def find_result_group_files(result_dir, result_basename):
    csg_files = sorted(glob.glob(f'{result_dir}/{result_basename}*_particles.csg'))
    assert len(csg_files) > 0, f'No result groups file found in {result_dir}'
    return csg_files


def main():
    args = cryopicls.args.autorefine_cryosparc.parse_args()

    csg_files = find_result_group_files(
        args.cryopicls_result_dir, args.cryopicls_result_basename)

    for csg_file in csg_files:
        # Instanciate communicator
        csparc_com = cryopicls.autorefine.cryosparc.CryoSPARCCom(
            args.ssh_user, args.ssh_host, args.ssh_port, args.csparc_user_email
        )
        # Should assert cryoSPARC version >= v3 here or inside CryoSPARCCom

        # Import result group
        job_uid = csparc_com.import_clustering_result_group(
            args.csparc_project_uid, args.csparc_workspace_uid, csg_file,
            cache_dir=args.cache_dir, lane=args.csparc_lane,
            title=f'Import of cryoPICLS clustering result : {csg_file}'
        )

        # Reconstruction without refinement (reconstruction only)
        job_uid = csparc_com.make_job(
            'homo_reconstruct',
            args.csparc_project_uid, args.csparc_workspace_uid,
            params={
                'refine_symmetry': args.csparc_refine_symmetry,
                'refine_gs_resplit': True
            },
            input_group_connects={
                'particles': f'{job_uid}.particles',
                'mask': f'{args.csparc_consensus_job_uid}.mask'
            },
            title=csg_file
        )
        csparc_com.enqueue_job(
            args.csparc_project_uid, job_uid, lane=args.csparc_lane
        )

        # Refinement
        job_uid = csparc_com.make_job(
            'homo_refine_new',
            args.csparc_project_uid, args.csparc_workspace_uid,
            params={
                'refine_symmetry': args.csparc_refine_symmetry,
                'refine_gs_resplit': True
            },
            input_group_connects={
                'particles': f'{job_uid}.particles',
                'volume': f'{job_uid}.volume'
            },
            title=csg_file
        )
        csparc_com.enqueue_job(
            args.csparc_project_uid, job_uid, lane=args.csparc_lane
        )


if __name__ == '__main__':
    main()
