package scm;

import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;

/**
 * ScmClient — v2: HTTP client for Jason Agent ↔ Python SCM Server communication.
 *
 * Handles JSON serialization and primitive JSON parsing.
 * Communicates with scm_server.py running on localhost:8008.
 */
public class ScmClient {
    public static String BASE = "http://127.0.0.1:8008";

    public static String postJson(String path, String json) throws IOException {
        URL url = URI.create(BASE + path).toURL();
        HttpURLConnection con = (HttpURLConnection) url.openConnection();
        con.setRequestMethod("POST");
        con.setRequestProperty("Content-Type", "application/json; charset=utf-8");
        con.setConnectTimeout(5000);
        con.setReadTimeout(10000);
        con.setDoOutput(true);

        try (OutputStream os = con.getOutputStream()) {
            os.write(json.getBytes(StandardCharsets.UTF_8));
        }

        int code = con.getResponseCode();
        InputStream is = (code >= 200 && code < 300) ? con.getInputStream() : con.getErrorStream();

        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) sb.append(line);
        }

        if (code < 200 || code >= 300) {
            throw new IOException("HTTP Error " + code + ": " + sb.toString());
        }
        return sb.toString();
    }

    public static double getNumber(String json, String key) {
        String k = "\"" + key + "\"";
        int i = json.indexOf(k);
        if (i < 0) return 0.0;

        int c = json.indexOf(':', i);
        if (c < 0) return 0.0;
        int j = c + 1;
        while (j < json.length() && " \t\n\r\"".indexOf(json.charAt(j)) >= 0) j++;

        // Handle null
        if (j + 3 < json.length() && json.substring(j, j + 4).equals("null")) return 0.0;

        int end = j;
        while (end < json.length() && "0123456789-+.eE".indexOf(json.charAt(end)) >= 0) end++;

        try { return Double.parseDouble(json.substring(j, end)); }
        catch (Exception e) { return 0.0; }
    }

    public static String getString(String json, String key) {
        String k = "\"" + key + "\"";
        int i = json.indexOf(k);
        if (i < 0) return "";

        int c = json.indexOf(':', i);
        if (c < 0) return "";

        // Skip whitespace after colon
        int j = c + 1;
        while (j < json.length() && " \t\n\r".indexOf(json.charAt(j)) >= 0) j++;

        // Handle null
        if (j + 3 < json.length() && json.substring(j, j + 4).equals("null")) return "";

        int start = json.indexOf('\"', c + 1) + 1;
        int end = json.indexOf('\"', start);
        if (start > 0 && end > start) return json.substring(start, end);
        return "";
    }

    public static boolean getBoolean(String json, String key) {
        String k = "\"" + key + "\"";
        int i = json.indexOf(k);
        if (i < 0) return false;

        int c = json.indexOf(':', i);
        if (c < 0) return false;
        int j = c + 1;
        while (j < json.length() && " \t\n\r\"".indexOf(json.charAt(j)) >= 0) j++;

        if (json.startsWith("true", j)) return true;
        if (json.startsWith("false", j)) return false;
        if (j < json.length() && json.charAt(j) == '1') return true;
        return false;
    }
}
