package client;

import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.util.concurrent.ThreadLocalRandom;

public class PeerClient {
    private static final String HOST = "37.187.198.176";
    private static final int PORT = 1234;
    private static final int BUF_SIZE = 4096;
    private static final int MAX_TIME = 100000; //  1min 40sec: enough to get +300 upstream packets

    private static void request(Socket socket, String path) throws Exception {
        OutputStream out = socket.getOutputStream();
        InputStream in = socket.getInputStream();
        byte[] message = String.format("GET %s HTTP/1.1\r\n\r\n", path).getBytes();
        out.write(message, 0, message.length);
        out.flush();

        int n;
        byte[] buffer = new byte[BUF_SIZE];
        while ((n = in.read(buffer, 0, buffer.length)) != -1) {
        }

    }

    public static void main(String[] args) throws Exception {
        String[] files = new String[]{"large", "small", "none"};
        long init = System.currentTimeMillis(), fin;
        int requestNum = 0;

        do {
            Socket socket = new Socket(HOST, PORT);
            String file = "/Files/" + files[ThreadLocalRandom.current().nextInt(files.length)];
            System.out.println((++requestNum) + ": Requesting " + file);
            request(socket, file);
            socket.close();

            fin = System.currentTimeMillis();
        } while (fin - init < MAX_TIME);

        System.out.printf("%d requests, done in %s!\n", requestNum, fin - init);

    }
}
