import subprocess
import os
import time
import shutil
import re

import cryopicls


class CryoSPARCCom:
    def __init__(self, ssh_user, ssh_host, ssh_port, csparc_user_email, print_com=True,
                 sleep_time=1):
        self.ssh_user = ssh_user
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.csparc_user_email = csparc_user_email
        self.sleep_time = sleep_time
        self.print_com = print_com
        self.csparc_user_id = self.get_user_id()

    def sshcom(self, command):
        com = f"""ssh {self.ssh_user}@{self.ssh_host} -p {self.ssh_port} "{command}" """
        if self.print_com:
            print(com)

        ret = subprocess.run(com, shell=True, capture_output=True)

        assert ret.returncode == 0, ret.stderr.decode()

        ret_stdout = ret.stdout.decode().rstrip()

        return ret_stdout

    def get_user_id(self):
        com = f"""cryosparcm cli \\"GetUser('{self.csparc_user_email}')['_id']\\" """
        user_id = self.sshcom(com)
        return user_id

    def make_workspace(self, project_uid, title='', desc=''):
        com = f"""cryosparcm cli \\"create_empty_workspace(project_uid='{project_uid}', created_by_user_id='{self.csparc_user_id}'"""

        if title != '':
            com += f", title='{title}'"
        if desc != '':
            com += f", desc='{desc}'"

        com += ')\\"'

        workspace_uid = self.sshcom(com)
        time.sleep(self.sleep_time)

        return workspace_uid

    def make_job(self, job_type, project_uid, workspace_uid, params=None,
                 input_group_connects=None, title=''):
        if params is None:
            params = dict()

        com = f"""cryosparcm cli \\"make_job(job_type='{job_type}', title='{title}', project_uid='{project_uid}', workspace_uid='{workspace_uid}', user_id='{self.csparc_user_id}', params={params}"""

        if input_group_connects is not None:
            com += f", input_group_connects={input_group_connects}"

        com += ')\\"'

        job_uid = self.sshcom(com)
        time.sleep(self.sleep_time)

        return job_uid

    def enqueue_job(self, project_uid, job_uid, lane='default'):
        com = f"""cryosparcm cli \\"enqueue_job('{project_uid}', '{job_uid}', '{lane}')\\" """
        self.sshcom(com)
        time.sleep(self.sleep_time)

    def _transfer_result_group_to_cache(self, csg_file, cache_dir):
        assert os.path.exists(csg_file)
        cs_file, passthrough_file = \
            cryopicls.data_handling.cryosparc.get_metafiles_from_csg(csg_file)
        assert os.path.exists(cs_file)
        if passthrough_file is not None:
            assert os.path.exists(passthrough_file)

        os.makedirs(cache_dir, exist_ok=True)
        csg_file = shutil.copy2(csg_file, cache_dir)
        cs_file = shutil.copy2(cs_file, cache_dir)
        if passthrough_file is not None:
            passthrough_file = shutil.copy2(passthrough_file, cache_dir)

        return csg_file, cs_file, passthrough_file

    def wait_job_complete(self, project_uid, job_uid, sleep_time=1, print_msg=True):
        # wait_job_complete deplecated in v3?? not working..
        # com = f"""cryosparcm cli \\"wait_job_complete('{project_uid}', '{job_uid}')\\" """
        # self.sshcom(com)
        com = f"""cryosparcm cli \\"get_job('{project_uid}', '{job_uid}', 'status')\\" """
        if print_msg:
            print(f'Waiting job {project_uid}-{job_uid} for complete...')
        while True:
            ret = self.sshcom(com)
            m = re.match(r".+'status': '([a-z]+)'.*", ret)
            status = m.group(1)
            if status == 'completed':
                break
            time.sleep(sleep_time)
        return status

    def import_clustering_result_group(self, project_uid, workspace_uid,
                                       csg_file, cache_dir=None, title='',
                                       lane='default'):
        assert os.path.exists(csg_file)

        if cache_dir is not None:
            csg_file, cs_file, passthrough_file = \
                self._transfer_result_group_to_cache(csg_file, cache_dir)
        else:
            cs_file, passthrough_file = \
                cryopicls.data_handling.cryosparc.get_metafiles_from_csg(csg_file)

        job_uid = self.make_job(
            'import_result_group', project_uid, workspace_uid,
            params={'blob_path': csg_file}, title=title
        )

        self.enqueue_job(project_uid, job_uid, lane)

        self.wait_job_complete(project_uid, job_uid)

        return job_uid
