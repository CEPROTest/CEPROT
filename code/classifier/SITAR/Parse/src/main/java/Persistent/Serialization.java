package Persistent;


import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;


public class Serialization {

    private static ObjectMapper mapper =new ObjectMapper();

    public static String ObjToJSON(Object object) throws JsonProcessingException {
        return mapper.writeValueAsString(object);
    }

    public static <T> T json2BeanByType(String jsonStr, TypeReference tr) throws JsonProcessingException {
        return (T) mapper.readValue(jsonStr,tr);
    }
    public static <T> T json2Bean(String jsonStr,Class<T> clazz) throws JsonProcessingException {
        return mapper.readValue(jsonStr,clazz);
    }
}
