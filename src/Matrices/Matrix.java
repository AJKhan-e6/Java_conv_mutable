package Matrices;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Matrix{

private static Map<String, List<Object>> dataMap = new HashMap<>();

public static void main(String[] args) {
    readParquetFile("Resources/myFile0.parquet");
}

private static void readParquetFile(String filePath) {
    Path path = new Path(filePath);
    Configuration configuration = new Configuration();

    try (ParquetReader<GenericRecord> reader = AvroParquetReader
        .<GenericRecord>builder(HadoopInputFile.fromPath(path, configuration))
        .withConf(configuration)
        .build()) {

        GenericRecord record;
        while ((record = reader.read()) != null) {
            processRecord(record);
        }
        INDArray columnArray = convertToCollectiveNDArray();

        // Add a way to now stack these arrays in together with vstack so that a table/matrix is formed
        // Also maybe its better to just create a matrix which is the shape of the incoming data

    } catch (IOException e) {
        e.printStackTrace();
    }
}




private static void processRecord(GenericRecord record) {
    for (Schema.Field field : record.getSchema().getFields()) {
        Object value = record.get(field.name());
        Schema fieldSchema = field.schema();
        Schema.Type type = fieldSchema.getType();

        // Handle primitive types
        Object convertedValue = convertPrimitive(value, fieldSchema);
        System.out.print("[Field: " + field.name() + " - Value: " + convertedValue + "]\t");

        // Accumulate values based on field name
        if (convertedValue != null) {
            dataMap.computeIfAbsent(field.name(), k -> new ArrayList<>()).add(convertedValue);
        }
    }
    System.out.println();
}

private static Object convertPrimitive(Object value, Schema schema) {
    if (value == null) {
        return null;
    }

    return switch (schema.getType())
    {
        case INT -> (Integer) value;
        case LONG -> (Long) value;
        case FLOAT -> (Float) value;
        case DOUBLE -> (Double) value;
        case BOOLEAN -> (Boolean) value;
        case STRING -> value.toString();
        case BYTES -> ((ByteBuffer) value).array();
        case FIXED ->
            // Handle fixed type, typically for binary data
            ((GenericData.Fixed) value).bytes();
        case ENUM ->
            // Handle enum type, return as String
            value.toString();
        case NULL -> null; // Handle null type
        default -> value; // Fallback for unrecognized types
    };
}

private static void processStruct(GenericRecord record) {
    // Process each field of the struct
    record.getSchema().getFields().forEach(field -> {
        Object value = record.get(field.name());
        System.out.print("[Struct Field: " + field.name() + " - Value: " + value + "]\t");
    });
}

private static void processArray(GenericData.Array<?> array) {
    // Process each element of the array
    array.forEach(element -> System.out.println("[Array Element: " + element + "]\t"));
}

private static void processMap(Map<?, ?> map) {
    // Process each key-value pair of the map
    map.forEach((key, value) -> System.out.println("[Map Entry: Key = " + key + ", Value = " + value + "]\t"));
}



private static INDArray convertToCollectiveNDArray() {
    List<INDArray> columnArrays = new ArrayList<>();
    for (Map.Entry<String, List<Object>> entry : dataMap.entrySet()) {
        String key = entry.getKey();
        List<Object> values = entry.getValue();

        if (!values.isEmpty()) {
            INDArray columnArray = Nd4j.create(values.stream().mapToDouble(v -> ((Number) v).doubleValue()).toArray());
            columnArrays.add(columnArray);
        }
    }

    // Stack column arrays horizontally to create a collective ND4J matrix
    return Nd4j.hstack(columnArrays.toArray(new INDArray[0]));
}


}