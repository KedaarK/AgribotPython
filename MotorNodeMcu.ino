#ifdef ESP8266
#include <ESP8266WiFi.h>
#include <ESP8266mDNS.h>
#include <ESPAsyncTCP.h>
#elif defined(ESP32)
#include <WiFi.h>
#include <ESPmDNS.h>
#include <AsyncTCP.h>
#else
#error "Board not found"
#endif

#include <ESPAsyncWebServer.h>
#include <FirebaseESP8266.h>  // Firebase library for ESP8266
#include <DHT.h>

// =================== WIFI & FIREBASE CONFIG ===================
const char* ssid       = "Kedaar";
const char* password   = "Falked01";
#define FIREBASE_HOST  "agribot-1fbbc-default-rtdb.firebaseio.com"
#define FIREBASE_AUTH  "AIzaSyBX4Dxu3N7RAK2Nio_zVVmkN4SuP8wH1bc"

FirebaseData firebaseData;
FirebaseConfig firebaseConfig;

// =================== DHT SENSOR CONFIG ===================
#define DHTPIN D7       // <-- CHOOSE a pin that is free for DHT
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

// =================== MOTOR DRIVER (L298N) PINS ===================
// IN1, IN2, IN3, IN4 for direction
#define IN1 D1
#define IN2 D2
#define IN3 D3
#define IN4 D4

// ENA, ENB for enabling motors (speed control if PWM needed)
#define ENA D5
#define ENB D6

// =================== WEB SERVER ===================
AsyncWebServer server(80);

void notFound(AsyncWebServerRequest *request) {
  request->send(404, "text/plain", "Page Not Found");
}

// Motor functions
void forwardMotor() {
  // Example "full speed" forward on both sides
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  // If using PWM, do analogWrite(ENA, <0-1023>), etc.
}

void backwardMotor() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void leftMotor() {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void rightMotor() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void stopMotors() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

// =================== SETUP ===================
void setup() {
  Serial.begin(115200);

  // Set motor pins as output
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  // Set ENA, ENB as output and enable them (HIGH) if not using PWM
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  digitalWrite(ENA, HIGH);
  digitalWrite(ENB, HIGH);

  // Initialize motors off
  stopMotors();

  // DHT sensor init
  dht.begin();

  // =========== CONNECT TO WIFI ===========
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected to Wi-Fi. IP Address:");
  Serial.println(WiFi.localIP());

  // =========== FIREBASE INIT ===========
  firebaseConfig.database_url = FIREBASE_HOST;
  firebaseConfig.signer.tokens.legacy_token = FIREBASE_AUTH;
  Firebase.begin(&firebaseConfig, nullptr); 
  Serial.println("Firebase Initialized.");

  // =========== ASYNC WEB SERVER ROUTES (OPTIONAL) ===========
  server.on("/", HTTP_GET, [](AsyncWebServerRequest *request) {
    String html = R"rawliteral(
    <!DOCTYPE html>
    <html>
    <head>
      <title>Agribot Control</title>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
    </head>
    <body style="text-align:center;">
      <h1>Agribot Motor Control</h1>
      <button onclick="location.href='/forward'">Forward</button>
      <button onclick="location.href='/backward'">Backward</button>
      <button onclick="location.href='/left'">Left</button>
      <button onclick="location.href='/right'">Right</button>
      <button onclick="location.href='/stop'">Stop</button>
      <br/><br/>
      <p>Check Firebase for sensor data updates.</p>
    </body>
    </html>
    )rawliteral";
    request->send(200, "text/html", html);
  });

  // Motor endpoints
  server.on("/forward", HTTP_GET, [](AsyncWebServerRequest *request) {
    Serial.println("Motor forward command via web");
    forwardMotor();
    request->redirect("/");
  });
  server.on("/backward", HTTP_GET, [](AsyncWebServerRequest *request) {
    Serial.println("Motor backward command via web");
    backwardMotor();
    request->redirect("/");
  });
  server.on("/left", HTTP_GET, [](AsyncWebServerRequest *request) {
    Serial.println("Motor left command via web");
    leftMotor();
    request->redirect("/");
  });
  server.on("/right", HTTP_GET, [](AsyncWebServerRequest *request) {
    Serial.println("Motor right command via web");
    rightMotor();
    request->redirect("/");
  });
  server.on("/stop", HTTP_GET, [](AsyncWebServerRequest *request) {
    Serial.println("Motor stop command via web");
    stopMotors();
    request->redirect("/");
  });

  server.onNotFound(notFound);
  server.begin();
  Serial.println("HTTP server started on port 80");
}

// =================== LOOP ===================
void loop() {
  // 1) Read temperature/humidity from DHT
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  // 2) Check if data is available on the serial port from Arduino
  if (Serial.available()) {
    // We have new soil moisture & distance data from the Arduino
    String sensorData = Serial.readStringUntil('\n');  // Read until newline
    Serial.print("Data received from Arduino: ");
    Serial.println(sensorData);  // Print the raw sensor data

    // e.g., the format: "Moisture:600,Distance:120" (example)
    // Split the string at the comma
    int commaIndex = sensorData.indexOf(',');
    if (commaIndex != -1) {
      String moistureData = sensorData.substring(sensorData.indexOf(':') + 1, commaIndex);
      String distanceData = sensorData.substring(commaIndex + 1); // e.g. "Distance:120"

      int soilMoistureValue = moistureData.toInt();
      int colonIndex = distanceData.indexOf(':');
      float distanceValue = 0.0;
      if (colonIndex != -1) {
        distanceValue = distanceData.substring(colonIndex + 1).toFloat();
      }

      Serial.print("Parsed Soil Moisture: ");
      Serial.println(soilMoistureValue);
      Serial.print("Parsed Distance: ");
      Serial.println(distanceValue);

      // 3) If distance is valid, push everything (incl. DHT) to Firebase
      if (distanceValue > 0) {
        bool successSoil = Firebase.setInt(firebaseData, "/sensors/soilMoisture", soilMoistureValue);
        bool successDist = Firebase.setFloat(firebaseData, "/sensors/distance", distanceValue);
        bool successTemp = Firebase.setFloat(firebaseData, "/sensors/temperature", temperature);
        bool successHum  = Firebase.setFloat(firebaseData, "/sensors/humidity", humidity);

        if (successSoil && successDist && successTemp && successHum) {
          Serial.println("Data (moisture, distance, temp, humidity) sent to Firebase successfully!");
        } else {
          Serial.print("Error sending data to Firebase: ");
          Serial.println(firebaseData.errorReason());
        }
      } else {
        Serial.println("Invalid distance value, not sending to Firebase.");
      }
    } else {
      Serial.println("Error: Data format is incorrect");
    }
  } 
  else {
    // ====================
    // NO NEW DATA from Arduino
    // ====================

    Serial.println("No data received from Arduino this loop.");

    // But we STILL want to upload DHT data (temperature & humidity) to Firebase
    bool successTemp = Firebase.setFloat(firebaseData, "/sensors/temperature", temperature);
    bool successHum  = Firebase.setFloat(firebaseData, "/sensors/humidity", humidity);

    if (successTemp && successHum) {
      Serial.println("DHT data (temp, humidity) sent to Firebase successfully!");
    } else {
      Serial.print("Error sending DHT data to Firebase: ");
      Serial.println(firebaseData.errorReason());
    }
  }

  delay(2000);  // Sample rate
}

