const int soilMoisturePin = A0;
const int trigPin = 2;
const int echoPin = 3;

void setup() {
  Serial.begin(115200); // Start the serial communication
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
}

void loop() {
  // Read soil moisture
  int soilMoistureValue = analogRead(soilMoisturePin);

  // Read ultrasonic sensor distance
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  long duration = pulseIn(echoPin, HIGH);
  float distance = duration * 0.034 / 2;

  // Send data to NodeMCU
  Serial.print("SoilMoisture:");
  Serial.print(soilMoistureValue);
  Serial.print(",Distance:");
  Serial.println(distance);  // Send the data over serial

  delay(3000);  // Delay for readability and to prevent overload
}
