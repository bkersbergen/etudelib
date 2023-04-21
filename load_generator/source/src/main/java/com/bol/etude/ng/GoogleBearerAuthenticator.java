package com.bol.etude.ng;

import com.google.auth.oauth2.GoogleCredentials;

import java.io.IOException;

public class GoogleBearerAuthenticator implements Authenticator {

    final GoogleCredentials creds;

    GoogleBearerAuthenticator() {
        try {
            creds = GoogleCredentials.getApplicationDefault();
            creds.refreshIfExpired();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String token() {
        return  creds.getAccessToken().getTokenValue();
    }
}
