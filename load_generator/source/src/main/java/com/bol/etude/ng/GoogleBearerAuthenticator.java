package com.bol.etude.ng;

import com.google.auth.oauth2.GoogleCredentials;

import java.io.IOException;

public class GoogleBearerAuthenticator implements Authenticator {

    final GoogleCredentials creds;

    GoogleBearerAuthenticator() {
        try {
            creds = GoogleCredentials.getApplicationDefault();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String token() {
        try {
            creds.refreshIfExpired();
            return  creds.getAccessToken().getTokenValue();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
