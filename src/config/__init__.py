import os
from pathlib import Path

SEED = 1337

DATA_PATH = os.path.join(Path(os.path.abspath(__file__)).parent.parent.parent, "data")

training_file_name = "training.jsonl"
validation_file_name = "validation.jsonl"

DEFAULT_COMPANY_BINS = {
    "Telecommunications": [
        "ATT", "AWSSupport", "AzureSupport", "BoostCare", "CenturyLinkHelp", "CoxHelp",
        "MTNC_Care", "O2", "Safaricom_Care", "TMobileHelp", "USCellularCares", "VerizonSupport", "airtel_care", "idea_cares"
    ],
    "Technology & Software": [
        "AdobeCare", "AmazonHelp", "AppleSupport", "AsurionCares", "DropboxSupport",
        "GooglePlayMusic", "HPSupport", "MicrosoftHelps", "NortonSupport", "OfficeSupport",
        "PandoraSupport", "XboxSupport", "YahooCare", "asksalesforce"
    ],
    "Financial Services": [
        "AskAmex", "AskCiti", "AskRBC", "AskRobinhood", "AskVirginMoney", "Ask_WellsFargo",
        "BofA_Help", "ChaseSupport", "SCsupport"
    ],
    "Retail & E-commerce": [
        "AldiUK", "ArgosHelpers", "AskTarget", "GreggsOfficial", "MOO", "Morrisons",
        "NeweggService", "Tesco", "Walmart", "marksandspencer", "sainsburys", "sizehelpteam"
    ],
    "Food & Beverage": [
        "ArbysCares", "AskPapaJohns", "AskTigogh", "ChipotleTweets", "DunkinDonuts",
        "JackBox", "KFC_UKI_Help", "McDonalds", "TacoBellTeam", "askpanera"
    ],
    "Travel & Hospitality": [
        "AirAsiaSupport", "AirbnbHelp", "AlaskaAir", "AmericanAir", "British_Airways",
        "Delta", "HiltonHelp", "IHGService", "JetBlue", "Kimpton", "LondonMidland",
        "SouthwestAir", "VirginAmerica", "VirginAtlantic", "VirginTrains", "nationalrailenq"
    ],
    "Transportation & Public Services": [
        "GWRHelp", "Postmates_Help", "SW_Help", "TfL", "Uber_Support", "UPSHelp"
    ],
    "Entertainment & Streaming": [
        "ATVIAssist", "AskPlayStation", "SpotifyCares", "hulu_support"
    ],
    "Customer Support Services": [
        "AskeBay", "AskPayPal", "AskSeagate", "comcastcares", "mediatemplehelp"
    ],
    "Social Media & Online Services": [
        "TwitterSupport"
    ],
    "None": []
}

training_file_path = os.path.join(DATA_PATH, training_file_name)
validation_file_path = os.path.join(DATA_PATH, validation_file_name)
