from BidRequest import BidRequest
from Bidder import Bidder
import random

class Bid(Bidder):

    def __init__(self):

        #Initializes the bidder parameters.

        self.bidRatio = 50  # Ratio of bidding in percent
        self.fixedBidPrice = 300  # Fixed bid price
        # Other model-related variables can be defined here
        # self.model: dict[str, str] = {}
        
    def getBidPrice(self, bidRequest : BidRequest) -> int:

        bidPrice = -1

        if random.randint(0, 99) < self.bidRatio:  # Equivalent to r.nextInt(100)
            bidPrice = self.fixedBidPrice

        return bidPrice