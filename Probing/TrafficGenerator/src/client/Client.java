package client;

import com.msopentech.thali.toronionproxy.Utilities;

import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;

public class Client {
    public static final String SERVER_HOST = "176.78.24.84";
    public static final int SERVER_PORT = 8080;

    public static final String TOR_HOST = "127.0.0.1";
    public static final int TOR_PORT = 9050;
    public static final int TOR_BUFFER_SIZE = 4096;

    private static void http() throws Exception {
        Socket clientSocket = new Socket(SERVER_HOST, SERVER_PORT);
        OutputStream out = clientSocket.getOutputStream();
        InputStream in = clientSocket.getInputStream();

        byte[] request = "GET / HTTP/1.1\r\n\r\n".getBytes();
        out.write(request, 0, request.length);
        out.flush();

        int n;
        byte[] buffer = new byte[TOR_BUFFER_SIZE];
        while ((n = in.read(buffer, 0, buffer.length)) != -1)
            System.out.write(buffer, 0, n);

    }

    private static void tor() throws Exception {
        Socket clientSocket = Utilities.socks4aSocketConnection(SERVER_HOST, SERVER_PORT, TOR_HOST, TOR_PORT);
        clientSocket.setReceiveBufferSize(TOR_BUFFER_SIZE);
        clientSocket.setSendBufferSize(TOR_BUFFER_SIZE);
        OutputStream out = clientSocket.getOutputStream();
        out.flush();

        out.write(String.format("GET %s HTTP/1.1\r\n\r\n", "/").getBytes());
        out.flush();

        InputStream in = clientSocket.getInputStream();
        int n;
        byte[] buffer = new byte[TOR_BUFFER_SIZE];
        StringBuffer content = new StringBuffer();
        while ((n = in.read(buffer, 0, buffer.length)) >= 0) {
            content.append(buffer);
        }
        clientSocket.close();
        System.out.println(content);
    }


    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.out.println("Usage: java Client <http/tor>");
            System.exit(0);
        }

        if (args[0].equals("http"))
            http();
        else tor();
    }
}
