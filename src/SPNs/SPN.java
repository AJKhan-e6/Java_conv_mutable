package SPNs;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.io.PrintStream;

public class SPN {

    // Constants
    private static final double MIN_PERCENTAGE_SLICE = 0.10;
    private static final int MIN_INSTANCE_SLICE = 0; // Example value, adjust as needed
    private Node root;
    private int numRows;

    // Constructor to initialize the SPN
    public SPN(Node root) {
        this.root = root;
        this.numRows = 0;
    }

    // Update the SPN with a row
    public void update(int[] row, UpdateType updateType) {
        SmallBitset variables = new SmallBitset((1 << row.length) - 1);
        root.update(row, variables, updateType);
    }

    // Calculate the likelihood of the SPN given a filter
    public double likelihood(Filter filter) {
        return root.evaluate(filter, filter.begin().first, EvaluationType.APPROXIMATE).second;
    }

    // Calculate the upper bound of the SPN given a filter
    public double upperBound(Filter filter) {
        return root.evaluate(filter, filter.begin().first, EvaluationType.UPPER_BOUND).second;
    }

    // Calculate the lower bound of the SPN given a filter
    public double lowerBound(Filter filter) {
        return root.evaluate(filter, filter.begin().first, EvaluationType.LOWER_BOUND).second;
    }

    // Calculate the expectation for a specific attribute and filter
    public double expectation(int attributeId, Filter filter) {
        Filter filterCopy = filter.copyWithExpectationOperation();
        Pair<Double, Double> result = root.evaluate(filterCopy, filterCopy.begin().first, EvaluationType.APPROXIMATE);
        double condExpectation = result.first;
        double likelihood = result.second;
        double llh = 1;

        if (!filter.isEmpty()) {
            if (likelihood == 0) {
                return 0;
            }
            llh = likelihood;
        }

        return condExpectation / llh;
    }

    // Update a row in the SPN
    public void updateRow(int[] oldRow, int[] updatedRow) {
        deleteRow(oldRow);
        insertRow(updatedRow);
    }

    // Insert a row into the SPN
    public void insertRow(int[] row) {
        update(row, UpdateType.INSERT);
        numRows++;
    }

    // Delete a row from the SPN
    public void deleteRow(int[] row) {
        update(row, UpdateType.DELETE);
        numRows--;
    }

    // Estimate the number of distinct values for an attribute
    public double estimateNumberDistinctValues(int attributeId) {
        return root.estimateNumberDistinctValues(attributeId);
    }

    // Dump the SPN structure to the console
    public void dump() {
        dump(System.err);
    }

    // Dump the SPN structure to a specified output stream
    public void dump(PrintStream out) {
        out.println();
        root.dump(out);
    }

    // Assume there's a class Product with a static method makeUnique
    static class Product {
        static Product makeUnique(List<ProductChildWithVariables> children, int numRows) {
            // Implementation of the makeUnique method
            // ...
            return new Product(); // Placeholder return
        }
    }

    // Assume there's a class ProductChildWithVariables
    static class ProductChildWithVariables {
        // Implementation of ProductChildWithVariables
        // ...
    }
    public class LearningData {
        public final Vector<Vector<Float>> data;
        public final Vector<Vector<Float>> normalized;
        public final Vector<Vector<Integer>> nullMatrix;
        public final SmallBitset variables;
        public final Vector<LeafType> leafTypes;

        public LearningData(
                Vector<Vector<Float>> data,
                Vector<Vector<Float>> normalized,
                Vector<Vector<Integer>> nullMatrix,
                SmallBitset variables,
                Vector<LeafType> leafTypes
        ) {
            this.data = new Vector<>(data);
            this.normalized = new Vector<>(normalized);
            this.nullMatrix = new Vector<>(nullMatrix);
            this.variables = variables;
            this.leafTypes = new Vector<>(leafTypes);
        }
    }

    private static class Variable {
        // Implement the Variable class based on your actual data structures
    }

    private static class Vector {
        // Implement the Vector class based on your actual data structures
    }

    private static class Matrix {
        // Implement the Matrix class based on your actual data structures
    }

    private static class Node {
        // Implement the Node class based on your actual data structures
    }

    private static class LeafType {
        // Implement the LeafType class based on your actual data structures
    }

    private static class PlaceholderNode extends Node {
        // Implement the PlaceholderNode class based on your actual data structures
    }

    // Add necessary imports and other classes as needed

    public static Product createProductMinSlice(LearningData ld) {
        // Initialize an empty list to store product children
        List<Product.ChildWithVariables> children = new ArrayList<>();

        // Get an iterator for variables in the learning data
        Iterator<Variable> variableIterator = ld.getVariables().iterator();

        // Loop over each column in the learning data
        for (int i = 0; i < ld.getData().cols(); i++) {
            // Extract data, normalized data, null matrix, and variables for the current column
            Vector data = ld.getData().col(i);
            Vector normalized = ld.getNormalized().col(i);
            Matrix nullMatrix = ld.getNullMatrix().col(i);
            Set<Variable> variables = variableIterator.next().asSet();

            // Create a list of leaf types for the current column
            List<LeafType> splitLeafTypes = List.of(ld.getLeafTypes().get(i));

            // Create a new LearningData instance for the current column
            LearningData splitData = new LearningData(data, normalized, nullMatrix, variables, splitLeafTypes);

            // Create a Product.ChildWithVariables with the learned node and variables
            Node learnedNode = learnNode(splitData);
            Product.ChildWithVariables child = new Product.ChildWithVariables(learnedNode, variables);

            // Add the child to the list of product children
            children.add(child);
        }

        // Create a Product node with the collected children and return it
        return new Product(children, ld.getData().rows());
    }

    static Product createProductRDC(LearningData ld, List<List<Integer>> columnCandidates, List<List<Integer>> variableCandidates) {
        List<ProductChildWithVariables> children = new ArrayList<>();

        // Loop over each current split
        for (int currentSplit = 0; currentSplit < columnCandidates.size(); currentSplit++) {
            List<Integer> currentSplitColumns = columnCandidates.get(currentSplit);
            int splitSize = currentSplitColumns.size();

            // Initialize lists to store split leaf types and column indices
            List<Integer> splitLeafTypes = new ArrayList<>(splitSize);
            List<Integer> columnIndex = new ArrayList<>(splitSize);

            // Loop over the columns in the current split
            for (int it : currentSplitColumns) {
                splitLeafTypes.add(ld.getLeafTypes().get(it));
                columnIndex.add(it);
            }

            // Extract data, normalized data, null matrix, and variables for the selected columns
            LearningData splitData = new LearningData(
                    ld.getData().getRows(columnIndex),
                    ld.getNormalized().getRows(columnIndex),
                    ld.getNullMatrix().getRows(columnIndex),
                    variableCandidates.get(currentSplit),
                    splitLeafTypes
            );

            // Create a Product::ChildWithVariables with the learned node and variables
            ProductChildWithVariables child = new ProductChildWithVariables(learnNode(splitData),
                    variableCandidates.get(currentSplit));

            // Add the child to the list of product children
            children.add(child);
        }

        // Create a Product node with the collected children and return it
        return Product.makeUnique(children, ld.getData().getRows());
    }

    public static Sum createSum(LearningData ld) {
        // Get the number of rows in the data
        int num_rows = ld.getData().rows();

        // Set the initial number of clusters to 2
        int k = 2;

        // Initialize variables for tracking the previous clustering results
        int prev_num_split_nodes = 0;
        List<List<Integer>> prev_cluster_row_ids = new ArrayList<>();
        List<List<Integer>> prev_cluster_column_candidates = new ArrayList<>();
        List<List<Integer>> prev_cluster_variable_candidates = new ArrayList<>();
        Matrix prev_centroids = new Matrix(0, 0); // Define your Matrix class

        // Continue the loop until a good clustering is achieved
        while (true) {
            // Initialize the count of split nodes in this iteration
            int num_split_nodes = 0;

            // Perform K-Means clustering and get labels and centroids
            Pair<int[], Matrix> kMeansResult = kmeansWithCentroids(ld.getNormalized(), k);
            int[] labels = kMeansResult.getFirst();
            Matrix centroids = kMeansResult.getSecond();

            // Initialize lists to store clustering results for each cluster
            List<List<Integer>> cluster_row_ids = new ArrayList<>();
            List<List<Integer>> cluster_column_candidates = new ArrayList<>();
            List<List<Integer>> cluster_variable_candidates = new ArrayList<>();

            // Assign data points to clusters
            for (int current_row = 0; current_row < num_rows; current_row++) {
                cluster_row_ids.get(labels[current_row]).add(current_row);
            }

            // Check if there's only one cluster
            boolean only_one_cluster = !cluster_row_ids.get(0).isEmpty();
            for (int i = 1; i < k; i++) {
                only_one_cluster = only_one_cluster && cluster_row_ids.get(i).isEmpty();
            }

            // If there's only one cluster, split it into k clusters
            if (only_one_cluster) {
                int subvector_size = cluster_row_ids.get(0).size() / k;
                List<List<Integer>> new_cluster_row_ids = new ArrayList<>();
                for (int cluster_id = 0; cluster_id < k - 1; cluster_id++) {
                    List<Integer> subvector = cluster_row_ids.get(0).subList(
                            cluster_id * subvector_size,
                            (cluster_id + 1) * subvector_size
                    );
                    new_cluster_row_ids.add(subvector);
                }
                new_cluster_row_ids.set(k - 1, cluster_row_ids.get(0).subList((k - 1) * subvector_size, cluster_row_ids.get(0).size()));
                cluster_row_ids = new_cluster_row_ids;
            }

            // Iterate over clusters
            for (int label_id = 0; label_id < k; label_id++) {
                int cluster_size = cluster_row_ids.get(label_id).size();

                // Skip empty clusters
                if (cluster_size == 0) {
                    continue;
                }

                // Extract data for the current cluster
                Matrix data = ld.getData().getRows(cluster_row_ids.get(label_id));

                // Check if the cluster size is small
                if (cluster_size <= MIN_INSTANCE_SLICE) {
                    num_split_nodes++;
                    cluster_column_candidates.add(new ArrayList<>());
                    cluster_variable_candidates.add(new ArrayList<>());
                } else {
                    // Perform the RDC split
                    Pair<List<Integer>, List<Integer>> rdcSplitResult = rdcSplit(data, ld.getVariables());
                    List<Integer> current_column_candidates = rdcSplitResult.getFirst();
                    List<Integer> current_variable_candidates = rdcSplitResult.getSecond();

                    if (current_column_candidates.size() > 1) {
                        num_split_nodes++;
                    }

                    cluster_column_candidates.add(current_column_candidates);
                    cluster_variable_candidates.add(current_variable_candidates);
                }
            }

            // Check stopping conditions
            if ((num_split_nodes <= prev_num_split_nodes || prev_num_split_nodes == prev_cluster_row_ids.size()) && prev_num_split_nodes != 0) {
                List<Sum.ChildWithWeight> children = new ArrayList<>();

                // Iterate over clusters and create children
                for (int cluster_id = 0; cluster_id < k - 1; cluster_id++) {
                    Matrix data = ld.getData().getRows(prev_cluster_row_ids.get(cluster_id));
                    Matrix normalized = ld.getNormalized().getRows(prev_cluster_row_ids.get(cluster_id));
                    Matrix null_matrix = ld.getNullMatrix().getRows(prev_cluster_row_ids.get(cluster_id));
                    LearningData cluster_data = new LearningData(data, normalized, null_matrix, ld.getVariables(), ld.getLeafTypes());
                    double weight = data.rows() / (double) num_rows;
                    int cluster_vertical_partitions = prev_cluster_column_candidates.get(cluster_id).size();

                    // Determine child node type based on cluster characteristics
                    Node child_node;
                    if (cluster_vertical_partitions == 0) {
                        child_node = createProductMinSlice(cluster_data);
                    } else if (cluster_vertical_partitions == 1) {
                        child_node = createSum(cluster_data);
                    } else {
                        child_node = createProductRDC(cluster_data,
                                prev_cluster_column_candidates.get(cluster_id),
                                prev_cluster_variable_candidates.get(cluster_id));
                    }

                    Sum.ChildWithWeight child = new Sum.ChildWithWeight(child_node, weight, prev_centroids.row(cluster_id));
                    children.add(child);
                }

                // Create a Sum node with the collected children and return it
                return new Sum(children, num_rows);
            }

            // Update previous clustering results
            prev_num_split_nodes = num_split_nodes;
            prev_cluster_row_ids = cluster_row_ids;
            prev_cluster_column_candidates = cluster_column_candidates;
            prev_cluster_variable_candidates = cluster_variable_candidates;
            prev_centroids = centroids;

            // Increment the number of clusters
            k++;
        }
    }

    // Other methods and helper functions (kmeansWithCentroids, rdcSplit, createProductMinSlice, createSum, createProductRDC) go here
    // ...

    public static Leaf learnNode(LearningData ld) {
        // Get the number of rows and columns in the data
        int numRows = ld.getData().rows();
        int numCols = ld.getData().cols();

        // Build a leaf node
        if (numCols == 1) {
            if (ld.getLeafTypes()[0] == LeafType.DISCRETE) {
                List<Bin> bins = new ArrayList<>();
                int nullCounter = 0;

                // Check if all values are NULL
                if (ld.getNullMatrix().min() == 1) {
                    return new DiscreteLeaf(Collections.emptyList(), 1.0);
                }

                for (int i = 0; i < numRows; i++) {
                    // Check if the value is NULL
                    if (ld.getNullMatrix().get(i, 0) == 1) {
                        nullCounter++;
                        continue;
                    }

                    double currentValue = ld.getData().get(i, 0);
                    int lowerBound = binarySearch(bins, currentValue);

                    if (lowerBound == bins.size()) {
                        bins.add(new Bin(currentValue, 1));
                    } else if (bins.get(lowerBound).getValue() == currentValue) {
                        bins.get(lowerBound).incrementCumulativeProbability();
                    } else {
                        bins.add(lowerBound, new Bin(currentValue, 1));
                    }
                }

                bins.get(0).setCumulativeProbability(bins.get(0).getCumulativeProbability() / numRows);
                for (int i = 1; i < bins.size(); i++) {
                    bins.get(i).setCumulativeProbability(bins.get(i).getCumulativeProbability() / numRows
                            + bins.get(i - 1).getCumulativeProbability());
                }

                double nullProbability = (double) nullCounter / numRows;
                return new DiscreteLeaf(bins, nullProbability);
            } else {
                List<Bin> bins = new ArrayList<>();
                int nullCounter = 0;

                // Check if all values are NULL
                if (ld.getNullMatrix().min() == 1) {
                    return new ContinuousLeaf(Collections.emptyList(), 1.0, 1.0);
                }

                double max = ld.getData().max();
                double min = ld.getData().min();
                int numBins = 1 + log2Ceil(numRows);
                double binWidth = (max - min) / numBins;
                double lowerBound = min;
                int lowerBoundCounter = 0;

                bins = initializeBins(numBins, min, binWidth);

                for (int i = 0; i < numRows; i++) {
                    // Check if the value is NULL
                    if (ld.getNullMatrix().get(i, 0) == 1) {
                        nullCounter++;
                        continue;
                    }

                    double currentValue = ld.getData().get(i, 0);

                    if (currentValue == lowerBound) {
                        lowerBoundCounter++;
                        continue;
                    }

                    int stdLowerBound = binarySearch(bins, currentValue);

                    if (stdLowerBound != bins.size()) {
                        bins.get(stdLowerBound).incrementCumulativeProbability();
                    }
                }

                double lowerBoundProbability = (double) lowerBoundCounter / numRows;
                bins.get(0).setCumulativeProbability(bins.get(0).getCumulativeProbability() / numRows);
                for (int i = 1; i < bins.size(); i++) {
                    bins.get(i).setCumulativeProbability(bins.get(i).getCumulativeProbability() / numRows
                            + bins.get(i - 1).getCumulativeProbability());
                }

                double nullProbability = (double) nullCounter / numRows;
                return new ContinuousLeaf(bins, lowerBound, lowerBoundProbability, nullProbability);
            }
        }

        // Build a product node with the minimum instance slice
        if (numRows <= MIN_INSTANCE_SLICE) {
            return createProductMinSlice(ld);
        }

        // Build a product node with the RDC algorithm
        List<Integer> columnCandidates = new ArrayList<>();
        List<String> variableCandidates = new ArrayList<>();
        rdcSplit(ld.getData(), ld.getVariables(), columnCandidates, variableCandidates);

        if (columnCandidates.size() != 1) {
            return createProductRdc(ld, columnCandidates, variableCandidates);
        }

        // Build a sum node
        return createSum(ld);
    }

    public static SPN learnSPN(Matrix data, Matrix nullMatrix, List<LeafType> leafTypes) {
        // Get the number of rows in the data
        int numRows = data.rows();

        // Calculate MIN_INSTANCE_SLICE as 10% of the number of rows or 1, whichever is greater
        int MIN_INSTANCE_SLICE = Math.max((int) (MIN_PERCENTAGE_SLICE * numRows), 1);

        // If there are no rows in the data
        if (numRows == 0) {
            List<Bin> bins = new ArrayList<>();
            DiscreteLeaf discreteLeaf = new DiscreteLeaf(bins, 0, 0, 0);
            return new SPN(0, discreteLeaf);
        }

        // Replace NULL values in the data matrix with attribute means
        for (int colId = 0; colId < data.columns(); colId++) {
            // Check if there are no NULL values in the column
            if (nullMatrix.columnMax(colId) == 0) {
                continue; // No NULL values in the column, move to the next
            }

            // Check if there are only NULL values in the column
            if (nullMatrix.columnMin(colId) == 1) {
                continue; // Only NULL values in the column, move to the next
            }

            // Calculate the mean of non-NULL values in the column
            double mean = 0;
            int numNotNull = 0;
            for (int rowId = 0; rowId < numRows; rowId++) {
                // Check if the value is NULL
                if (nullMatrix.get(rowId, colId) == 1) {
                    continue; // Skip NULL values
                }

                // Update the mean using an iterative mean calculation
                mean += (data.get(rowId, colId) - mean) / ++numNotNull;
            }

            // Replace NULL values in the column with the calculated mean
            for (int rowId = 0; rowId < numRows; rowId++) {
                if (nullMatrix.get(rowId, colId) == 1) {
                    data.set(rowId, colId, mean);
                }
            }
        }

        // Create a SmallBitset with all variables
        SmallBitset variables = new SmallBitset((1 << data.columns()) - 1);

        // Normalize the data using min-max scaling
        Matrix normalized = normalizeMinMax(data);

        // Create a LearningData object with data, normalized data, nullMatrix, variables, and leafTypes
        LearningData ld = createLearningData(data, normalized, nullMatrix, variables, leafTypes);

        // Create an SPN object with the number of rows and the learned SPN node
        return new SPN(numRows, learnNode(ld));
    }

    // Other methods and helper functions (normalizeMinMax, createLearningData, learnNode) go here
    // ...

    // Helper methods (binarySearch, log2Ceil, initializeBins, createProductMinSlice, createProductRdc, createSum)
    // ...

    // Define LeafType, DecisionData, Bin, DiscreteLeaf, ContinuousLeaf classes
    // ...
}
