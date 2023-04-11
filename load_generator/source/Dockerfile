ARG PARENT_IMAGE
FROM $PARENT_IMAGE

WORKDIR /java

COPY --chown=java:java ./target/etude_loadgen.jar ./

# local java user
RUN chown -R java:java /java
USER java

COPY --from=redboxoss/scuttle:latest /scuttle /bin/scuttle
ENTRYPOINT ["/bin/scuttle", "java", "-jar", "etude_loadgen.jar"]
