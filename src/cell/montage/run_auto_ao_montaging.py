
# from .AOAutomontagingPython.AutoAOMontaging import AutoAOMontaging
from .AOAutomontaging_master.AutoAOMontaging import AutoAOMontaging
import os

def run_auto_ao_montaging(input_dir: str, loc_file: str, montage_dir: str):
    """
    Functions that is able to run the Matlab GUI directly without opening it and
    having to click to setup the run

    :param input_dir: the path of the base directory where the images are
    :type input_dir: str
    :param loc_file: the name of the csv file with the informations of the images
                     to montage
    :type loc_file: str
    :param montage_dir: the montage directory where the montage files will be
                        saved in the end
    :type montage_dir: str
    """

    # my_AutoAOMontaging = AutoAOMontaging.initialize()
    AutoAOMontaging(montage_dir, os.path.join(montage_dir, loc_file), input_dir)
    # loc_file = os.path.join(montage_dir, loc_file)
    # my_AutoAOMontaging.AutoAOMontaging(montage_dir, loc_file, input_dir, nargout=0)

    # my_AutoAOMontaging.terminate()


# if __name__ == '__main__':
#     # 10 279
#     run_auto_ao_montaging(r'P:\AOSLO\_automation\_PROCESSED\Photoreceptors\Healthy\_Results\Subject105\Session496',
#                           'loc.csv',
#                           r'P:\AOSLO\_automation\_PROCESSED\Photoreceptors\Healthy\_Results\Subject105\Session496\montaged')
#     # run_auto_ao_montaging(r'P:\AOSLO\_automation\_PROCESSED\Photoreceptors\Healthy\_Results\test\Subject108\Session506',
#     #                       r'P:\AOSLO\_automation\_PROCESSED\Photoreceptors\Healthy\_Results\test\Subject108\Session506\montaged\loc.csv',
#     #                       r'P:\AOSLO\_automation\_PROCESSED\Photoreceptors\Healthy\_Results\test\Subject108\Session506\montaged')