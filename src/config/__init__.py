import os
from pathlib import Path

SEED = 1337

DATA_PATH = os.path.join(Path(os.path.abspath(__file__)).parent.parent.parent, 'data')

training_file_name = 'training.jsonl'
validation_file_name = 'validation.jsonl'

DEFAULT_COMPANIES = ['ATT', 'ATVIAssist', 'AWSSupport', 'AdobeCare', 'AirAsiaSupport', 'AirbnbHelp', 'AlaskaAir', 'AldiUK', 'AmazonHelp', 'AmericanAir', 'AppleSupport', 'ArbysCares', 'ArgosHelpers', 'AskAmex', 'AskCiti', 'AskDSC', 'AskLyft', 'AskPapaJohns', 'AskPayPal', 'AskPlayStation', 'AskRBC', 'AskRobinhood', 'AskSeagate', 'AskTarget', 'AskTigogh', 'AskVirginMoney', 'Ask_Spectrum', 'Ask_WellsFargo', 'AskeBay', 'AsurionCares', 'AzureSupport', 'BofA_Help', 'BoostCare', 'British_Airways', 'CenturyLinkHelp', 'ChaseSupport', 'ChipotleTweets', 'CoxHelp', 'DellCares', 'Delta', 'DoorDash_Help', 'DropboxSupport', 'DunkinDonuts', 'GWRHelp', 'GloCare', 'GoDaddyHelp', 'GooglePlayMusic', 'GreggsOfficial', 'HPSupport', 'HiltonHelp', 'IHGService', 'JackBox', 'JetBlue', 'KFC_UKI_Help', 'Kimpton', 'LondonMidland', 'MOO', 'MTNC_Care', 'McDonalds', 'MicrosoftHelps', 'Morrisons', 'NeweggService', 'NikeSupport', 'NortonSupport', 'O2', 'OfficeSupport', 'PandoraSupport', 'Postmates_Help', 'SCsupport', 'SW_Help', 'Safaricom_Care', 'SouthwestAir', 'SpotifyCares', 'TMobileHelp', 'TacoBellTeam', 'Tesco', 'TfL', 'TwitterSupport', 'UPSHelp', 'USCellularCares', 'Uber_Support', 'VMUcare', 'VerizonSupport', 'VirginAmerica', 'VirginAtlantic', 'VirginTrains', 'Walmart', 'XboxSupport', 'YahooCare', 'airtel_care', 'askpanera', 'asksalesforce', 'comcastcares', 'hulu_support', 'idea_cares', 'marksandspencer', 'mediatemplehelp', 'nationalrailenq', 'sainsburys', 'sizehelpteam', 'sprintcare']

training_file_path = os.path.join(DATA_PATH, training_file_name)
validation_file_path = os.path.join(DATA_PATH, validation_file_name)