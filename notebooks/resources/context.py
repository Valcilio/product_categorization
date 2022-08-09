import os
from pathlib import Path

class Context():

    def __init__(self, **kwargs):

        self.found_root = False
        self.cur_path = Path(__file__).parent

    def set_proj_path(self, **kwargs):
        '''Look for .proj_root to set location'''

        for depth in range(15):
            list_files = os.listdir(self.cur_path)

            # If we don't find .proj_root in list_files, look in the folder 
            # above
            if '.proj_root' not in list_files:
                self.cur_path /= '..'
                continue

            # Solve PROJ_DIR and change var found_root
            PROJ_DIR = self.cur_path.resolve()
            self.found_root = True
            break

        # Check if PROJ_DIR was found
        if not self.found_root:
            raise RuntimeError("Project root folder not found. Please check if "
                            "the file ** .proj_root ** exists."
            )

        return PROJ_DIR

    def create_log_fld(self, **kwargs):
        '''Create variable with LOG_FLD path'''

        PROJ_DIR = self.set_proj_path()
        return PROJ_DIR / 'logs'

    def reports_fld(self, **kwargs):
        '''Create variable with reports path'''

        PROJ_DIR = self.set_proj_path()
        return PROJ_DIR / 'reports'

    def data_proc_fld(self, **kwargs):
        '''Create variable with PROC_FLD path'''

        PROJ_DIR = self.set_proj_path()
        return PROJ_DIR / 'data/processed'

    def data_raw_fld(self, **kwargs):
        '''Create variable with RAW_FLD path'''

        PROJ_DIR = self.set_proj_path()
        return PROJ_DIR / 'data/raw'

    def models_fld(self, **kwargs):
        '''Create variable with RAW_FLD path'''

        PROJ_DIR = self.set_proj_path()
        return PROJ_DIR / 'models'

    def scalers_fld(self, **kwargs):
        '''Create variable with SCALERS path'''

        PROJ_DIR = self.set_proj_path()
        return PROJ_DIR / 'scalers'