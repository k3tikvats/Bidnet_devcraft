To run evaluation using the PYTHON module, follow below steps:
1) Replace the bid.py with participating team's bid.py file
2) In line no 117, edit the absolute path of the file "test.dataset.10L.txt"
    parser = TestDataParser('/Users/anuragpandey/Downloads/evaluation/test.dataset.10L.txt')
3) Run the below python commands one by one on the terminal / command prompt by editing the absolute path of evaluationApp.py
    python /Users/anuragpandey/Downloads/evaluation/Python/evaluationApp.py --budget_set 1
    python /Users/anuragpandey/Downloads/evaluation/Python/evaluationApp.py --budget_set 2
    python /Users/anuragpandey/Downloads/evaluation/Python/evaluationApp.py --budget_set 3
    
Sum the score for each run to get the final score for each team.


To run evaluation using the JAVA module, follow below steps:
1) Replace the bid.java with participating team's bid.java file
2) In line no 104, edit the absolute path of the file "test.dataset.10L.txt"
    private static final String TEST_DATA_FILE_PATH = "/Users/adhir/Downloads/OneDrive_1_11-02-2025/test.dataset.10L.txt";
3) Run the below  commands one by one on the terminal / command prompt 
	mvn clean package
	java -jar PATH_TO_JAR 1 ex: ( java -jar '/Users/adhir/Downloads/evaluation/java/target/bidding-1.0.0.jar' 1 )
    java -jar PATH_TO_JAR 2 ex: ( java -jar '/Users/adhir/Downloads/evaluation/java/target/bidding-1.0.0.jar' 2 )
    java -jar PATH_TO_JAR 3 ex: ( java -jar '/Users/adhir/Downloads/evaluation/java/target/bidding-1.0.0.jar' 3 )
    
Sum the score for each run to get the final score for each team.