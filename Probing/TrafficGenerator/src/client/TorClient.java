package client;

import com.msopentech.thali.toronionproxy.Utilities;

import java.io.InputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.util.concurrent.ThreadLocalRandom;

public class TorClient {
    private static final String REMOTE_HOST = "85.243.193.251";
    private static final int REMOTE_PORT = 1238;
    private static final int MAX_TIME = 300000; // 5 minutes: enough to get +300 upstream packets - main client

    private static final String TOR_HOST = "192.168.1.73";
    private static final int TOR_PORT = 9150;
    private static final int TOR_BUFFER_SIZE = 4096;

    private static void request(String path) throws Exception {
        Socket clientSocket = Utilities.socks4aSocketConnection(REMOTE_HOST, REMOTE_PORT, TOR_HOST, TOR_PORT);
        clientSocket.setReceiveBufferSize(TOR_BUFFER_SIZE);
        clientSocket.setSendBufferSize(TOR_BUFFER_SIZE);

        OutputStream out = clientSocket.getOutputStream();
        out.flush();

        out.write(String.format("GET %s HTTP/1.1\r\n\r\n", path).getBytes());
        out.flush();

        InputStream in = clientSocket.getInputStream();
        int n;
        byte[] buffer = new byte[TOR_BUFFER_SIZE];
        while ((n = in.read(buffer, 0, buffer.length)) >= 0) {
        }
        clientSocket.close();
    }

    public static void main(String[] args) throws Exception {
        String[] files = new String[]{"large", "small", "none"};
        long init = System.currentTimeMillis(), fin;
        int requestNum = 0;

        do {
            String file = "/Files/" + files[ThreadLocalRandom.current().nextInt(files.length)];
            System.out.println((++requestNum) + ": Requesting " + file);
            request(file);

            Thread.sleep(ThreadLocalRandom.current().nextInt(2000, 4000));
            fin = System.currentTimeMillis();
        } while (fin - init < MAX_TIME);

        System.out.printf("%d requests, done in %s!\n", requestNum, fin - init);
    }
}
