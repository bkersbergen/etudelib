ARG PARENT_IMAGE
FROM $PARENT_IMAGE

WORKDIR /java

COPY --chown=java:java ./target/etude_loadgen.jar ./

# local java user
RUN chown -R java:java /java
USER java

COPY --from=redboxoss/scuttle:latest /scuttle /bin/scuttle

ENV JAVA_TOOL_OPTIONS "-XX:InitialRAMPercentage=90.0 -XX:MinRAMPercentage=90.0 -XX:MaxRAMPercentage=90.0 -XX:ActiveProcessorCount=4"

ENTRYPOINT ["/bin/scuttle", "java", "-jar", "etude_loadgen.jar"]
