package com.dtu.hackathon.bidding;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Bid class that need to be completed by the contest participants This is the
 * only place to implement your bidding algorithm Note that the class name (Bid)
 * and package name (com.dtu.hackathon.bidding) are fixed to make a correct
 * submission.
 *
 */
public class Bid implements Bidder {

	/**
	 * Bidder parameters definition
	 */

	// ratio of bidding in percent
	private int bidRatio = 50;

	// Fixed bid price
	private int fixedBidPrice = 300;

	// Other model related variables may be defined here
	// private Map<String, String> model = new HashMap<String, String>();


	@Override
	public int getBidPrice(BidRequest bidRequest) {

		int bidPrice = -1;
		Random r = new Random();

		if (r.nextInt(100) < this.bidRatio)
			bidPrice = this.fixedBidPrice;

		return bidPrice;
	}

}