ARG PARENT_IMAGE
FROM $PARENT_IMAGE

COPY --from=redboxoss/scuttle:latest /scuttle /bin/scuttle

WORKDIR /java

COPY --chown=java:java load_generator/source/.mvn .mvn
COPY --chown=java:java load_generator/source/src src
COPY --chown=java:java load_generator/source/mvnw ./
COPY --chown=java:java load_generator/source/pom.xml ./
COPY --chown=java:java load_generator/source/.sdkmanrc ./

RUN chmod +x ./mvnw
RUN ./mvnw clean package -DskipTests
RUN mv ./target/etude_loadgen.jar .

RUN groupadd -r java && useradd -r -g java -u 1001 java --home /java
RUN chown -R java:java /java
USER java

ENV JAVA_TOOL_OPTIONS "-XX:InitialRAMPercentage=90.0 -XX:MinRAMPercentage=90.0 -XX:MaxRAMPercentage=90.0 -XX:ActiveProcessorCount=4"

ENTRYPOINT ["/bin/scuttle", "java", "-jar", "etude_loadgen.jar"]
