from enum import Enum, unique


@unique
class ImageModalities(Enum):
    CO = 'Confocal'
    CS = 'CalculatedSplit'
    OA = 'OA850nm'  # or 'OA850' or '850nm' or 'OA'
    DF = 'DarkField'


IMAGE_EXTENSIONS = ['png', 'tif', 'jpg']
COORDINATES_EXTENSIONS = ['csv', 'txt']
