package com.dtu.hackathon.bidding.src.main.java.com.dtu.hackathon.bidding;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class EvaluationApp {

    public static class TestData {
        BidRequest bidRequest;
        int bidPrice;
        int payingPrice;
        int clicks;
        int conversions;
        int weight;
    }

    public static class BudgetInfo {
        public double budget;
        public long logLines;

        public BudgetInfo(double budget, long logLines) {
            this.budget = budget;
            this.logLines = logLines;
        }
    }
    public static class TestDataParser {
        private final BufferedReader br;

        public TestDataParser(String filePath) throws IOException {
            this.br = new BufferedReader(new FileReader(filePath));
        }

        public TestData readNextTestData() throws IOException, NumberFormatException {
            String line = br.readLine();
            if (line == null) {
                return null; // End of file
            }

            // Split the line by tab delimiter
            String[] fields = line.split("\t");

            // Create a BidRequest object
            BidRequest bidRequest = new BidRequest();

            // Map fields to the object
            bidRequest.setBidId(fields[0]);
            bidRequest.setTimestamp(fields[1]);
            bidRequest.setVisitorId(fields[3]);
            bidRequest.setUserAgent(fields[4]);
            bidRequest.setIpAddress(fields[5]);
            bidRequest.setRegion(fields[6]);
            bidRequest.setCity(fields[7]);
            bidRequest.setAdExchange(fields[8]);
            bidRequest.setDomain(fields[9]);
            bidRequest.setUrl(fields[10]);
            bidRequest.setAnonymousURLID(fields[11]);
            bidRequest.setAdSlotID(fields[12]);
            bidRequest.setAdSlotWidth(fields[13]);
            bidRequest.setAdSlotHeight(fields[14]);
            bidRequest.setAdSlotVisibility(fields[15]);
            bidRequest.setAdSlotFormat(fields[16]);
            bidRequest.setAdSlotFloorPrice(fields[17]);
            bidRequest.setCreativeID(fields[18]);
            bidRequest.setAdvertiserId(fields[22]);
            bidRequest.setUserTags(fields[23]);


            TestData testData = new TestData();
            testData.bidRequest = bidRequest;
            testData.bidPrice = Integer.parseInt(fields[19]);
            testData.payingPrice = Integer.parseInt(fields[20]);
            testData.clicks = Integer.parseInt(fields[24]);
            testData.conversions = Integer.parseInt(fields[25]);
            testData.weight = getWeight(bidRequest.getAdvertiserId());

            return testData;
        }

        private int getWeight(String advertiserId) {
            switch (advertiserId) {
                case "1458":
                    return 0;
                case "3358":
                    return 2;
                case "3386":
                    return 0;
                case "3427":
                    return 0;
                case "3476":
                    return 10;
            }
            return 0;
        }

        public void close() throws IOException {
            br.close();
        }
    }

    private static final String TEST_DATA_FILE_PATH
            = "/Users/adhir/Downloads/OneDrive_1_11-02-2025/test.dataset.10L.txt";

    private static Map<String, BudgetInfo> initializeBudgets1() {
        Map<String, BudgetInfo> exhaustedBudgets = new HashMap<>();
        exhaustedBudgets.put("1458", new BudgetInfo(15000.0, 0L));
        exhaustedBudgets.put("3358", new BudgetInfo(16000.0, 0L));
        exhaustedBudgets.put("3386", new BudgetInfo(15500.0, 0L));
        exhaustedBudgets.put("3427", new BudgetInfo(16500.0, 0L));
        exhaustedBudgets.put("3476", new BudgetInfo(14500.0, 0L));
        return exhaustedBudgets;
    }

    private static Map<String, BudgetInfo> initializeBudgets2() {
        Map<String, BudgetInfo> exhaustedBudgets = new HashMap<>();
        exhaustedBudgets.put("1458", new BudgetInfo(5000.0, 0L));
        exhaustedBudgets.put("3358", new BudgetInfo(4800.0, 0L));
        exhaustedBudgets.put("3386", new BudgetInfo(3900.0, 0L));
        exhaustedBudgets.put("3427", new BudgetInfo(4500.0, 0L));
        exhaustedBudgets.put("3476", new BudgetInfo(4200.0, 0L));
        return exhaustedBudgets;
    }
    private static Map<String, BudgetInfo> initializeBudgets3() {
        Map<String, BudgetInfo> exhaustedBudgets = new HashMap<>();
        exhaustedBudgets.put("1458", new BudgetInfo(20000.0, 0L));
        exhaustedBudgets.put("3358", new BudgetInfo(2000.0, 0L));
        exhaustedBudgets.put("3386", new BudgetInfo(15000.0, 0L));
        exhaustedBudgets.put("3427", new BudgetInfo(1800.0, 0L));
        exhaustedBudgets.put("3476", new BudgetInfo(8000.0, 0L));
        return exhaustedBudgets;
    }
    public static void main(String[] args) {
        System.out.println("Starting the bidding process...");
        try {
            TestDataParser parser = new TestDataParser(TEST_DATA_FILE_PATH);
            Bid bid = new Bid();
            TestData testData;
            if (args.length == 0) {
                System.out.println("Please provide a command-line argument (1, 2, or 3).");
                return;
            }

            String option = args[0];
            Map<String, BudgetInfo> budgets;

            switch (option) {
                case "1":
                    budgets = initializeBudgets1();
                    break;
                case "2":
                    budgets = initializeBudgets2();
                    break;
                case "3":
                    budgets = initializeBudgets3();
                    break;
                default:
                    System.out.println("Invalid option. Please use 1, 2, or 3.");
                    return;
            }
            Map<String, Double> exhaustedBudgets = new HashMap<>();
            BudgetInfo budgetInfo;
            long score = 0L;
            long logLines = 0L;
            while (exhaustedBudgets.size() != budgets.size() && (testData = parser.readNextTestData()) != null) {
                logLines++;
                String advertiserId = testData.bidRequest.getAdvertiserId();
                budgetInfo = budgets.get(advertiserId);
                if (budgetInfo.budget <= 0) {
                    continue;
                }
                long startTime = System.nanoTime();
                int bidPrice = bid.getBidPrice(testData.bidRequest);

                long endTime = System.nanoTime();
                long duration = (endTime - startTime) / 1_000_000; // Convert to milliseconds
                if (duration > 5) {
                    System.out.println("WARNING: getBidPrice took " + duration + " ms");
                }

                /*if(testData.bidPrice <= testData.payingPrice) {
                    System.out.println("testData.bidPrice: " + testData.bidPrice
                            + ", testData.payingPrice: " + testData.payingPrice
                            + ", %: " + (testData.payingPrice-testData.bidPrice)*100.0/testData.bidPrice);
                }*/

                if (bidPrice > testData.payingPrice) {
                    score += testData.clicks + (long) testData.weight *testData.conversions;
                    budgetInfo.budget -= testData.payingPrice / 1_000.0;
                }
                budgetInfo.logLines = logLines;

                if (budgetInfo.budget <= 0) {
                    exhaustedBudgets.put(advertiserId, budgetInfo.budget);
                }
            }

            System.out.printf("Score: %,d, Evaluated log lines: %,d%n", score, logLines);
            System.out.println("Remaining Budgets:");
            for (Map.Entry<String, BudgetInfo> entry : budgets.entrySet()) {
                System.out.printf("Advertiser %s: %.3f, Evaluated log lines: %,d%n",
                        entry.getKey(), entry.getValue().budget, entry.getValue().logLines);
            }
            parser.close();
            System.out.println("Bidding process completed successfully.");
        } catch (IOException e) {
            System.out.println("Failed to create BidRequestParser");
            throw new RuntimeException(e);
        } catch (NumberFormatException e) {
            System.out.println("Failed to parse integer");
            throw new RuntimeException(e);
        }
    }
}
