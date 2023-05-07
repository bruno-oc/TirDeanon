package server;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Server {
    private final static int PORT = 8080;

    private static void sendResponse(Socket client, String status, String contentType, byte[] content) throws IOException {
        OutputStream clientOutput = client.getOutputStream();
        clientOutput.write(("HTTP/1.1 " + status + "\r\n").getBytes());
        clientOutput.write(("ContentType: " + contentType + "\r\n").getBytes());
        clientOutput.write("\r\n".getBytes());
        clientOutput.write(content);
        clientOutput.write("\r\n\r\n".getBytes());
        clientOutput.flush();
        client.close();
    }

    private static void processRequest(Socket client, String request) throws Exception {
        String[] requestsLines = request.split("\r\n");
        String[] requestLine = requestsLines[0].split(" ");
        String method = requestLine[0];
        String path = requestLine[1];

        String accessLog = String.format("Client %s, method %s, path %s",
                client.toString(), method, path);
        System.out.println("[Request] " + accessLog);

        if (path.equals("/"))
            path = "/index.html";

        if (method.equals("GET")) {
            Path filePath = Paths.get("Files" + path);
            if (Files.exists(filePath))
                sendResponse(client, "200 OK", Files.probeContentType(filePath), Files.readAllBytes(filePath));
            else {
                byte[] notFoundContent = "<html><body><h1>404 Not Found!</h1></body></html>".getBytes();
                sendResponse(client, "404 Not Found", "text/html", notFoundContent);
            }
        } else {
            byte[] errorMsg = "<html><body><h1>405 Method Not Allowed</h1></body></html>".getBytes();
            sendResponse(client, "405 Method Not Allowed", "text/html", errorMsg);
        }
    }

    private static void handleClient(Socket client) throws Exception {
        System.out.println("[INFO] Got new request:");
        BufferedReader br = new BufferedReader(new InputStreamReader(client.getInputStream()));

        StringBuilder requestBuilder = new StringBuilder();
        String line;
        while (!(line = br.readLine()).isBlank())
            requestBuilder.append(line).append("\r\n");

        processRequest(client, requestBuilder.toString());
    }

    public static void main(String[] args) {
        try (ServerSocket serverSocket = new ServerSocket(PORT)) {
            System.out.println("[INFO] Server ready at port " + PORT + ", waiting for request...");
            Socket client;
            while (true) {
                handleClient((client = serverSocket.accept()));
                client.close();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
