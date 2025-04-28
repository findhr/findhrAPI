async function encryptData(data) {
    // Fetch the public key from the backend
    const publicKeyPem = await fetchPublicKey();
    if (!publicKeyPem) {
        throw new Error("Failed to retrieve public key.");
    }

    // Convert PEM format to ArrayBuffer for Web Crypto API
    const publicKey = await window.crypto.subtle.importKey(
        "spki",
        base64ToArrayBuffer(publicKeyPem), // Converts PEM to ArrayBuffer
        { name: "RSA-OAEP", hash: "SHA-256" },
        false,
        ["encrypt"]
    );

    // Encode the data into a Uint8Array
    const encodedData = new TextEncoder().encode(JSON.stringify(data));

    // Encrypt the encoded data
    const encryptedData = await window.crypto.subtle.encrypt(
        { name: "RSA-OAEP" },
        publicKey,
        encodedData
    );

    // Convert ArrayBuffer to Base64 for easy transmission
    return arrayBufferToBase64(encryptedData);
}

async function fetchPublicKey() {
    try {
        const response = await fetch("http://127.0.0.1:5000/public_key", { cache: "no-store" });
        if (!response.ok) throw new Error("Failed to fetch public key");
        let pem = await response.text();
        return pem.replace(/-----BEGIN PUBLIC KEY-----/, "")
                  .replace(/-----END PUBLIC KEY-----/, "")
                  .replace(/\n/g, "")
                  .trim();
    } catch (error) {
        console.error("Error fetching public key:", error);
    }
}

// Helper functions to convert between Base64 and ArrayBuffer
function arrayBufferToBase64(buffer) {
    return btoa(String.fromCharCode(...new Uint8Array(buffer)));
}

function base64ToArrayBuffer(base64) {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
}

async function sendData() {
    const providerId = document.getElementById("providerId").value;
    const userId = document.getElementById("userId").value;

    if (!providerId || !userId) {
        alert("Please enter valid Provider ID and User ID.");
        return;
    }

    const gender = document.getElementById("gender").value;
    const age = document.getElementById("age").value;
    const disability = document.getElementById("disability").value;

    // Encode categorical values
    const genderEncoded = encodeGender(gender);
    const ageEncoded = encodeAge(age);
    const disabilityEncoded = disability === "Yes" ? 1 : 0;

    // Generate random values
    const random1 = Math.floor((Math.random() * 2 - 1) * Number.MAX_SAFE_INTEGER);
    const random2 = Math.floor((Math.random() * 2 - 1) * Number.MAX_SAFE_INTEGER);
    const random3 = Math.floor((Math.random() * 2 - 1) * Number.MAX_SAFE_INTEGER);

    // Construct JSON payload for Backend A and B
    const dataA = { providerId, userId, gender: genderEncoded - random1, age: ageEncoded - random2, disabled: disabilityEncoded - random3 };
    const dataB = { providerId, userId, gender: genderEncoded + random1, age: ageEncoded + random2, disabled: disabilityEncoded + random3 };

    try {
        const encryptedDataA = await encryptData(dataA);
        const encryptedDataB = await encryptData(dataB);

        await fetch("http://127.0.0.1:5000/store_service_provider", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ data: encryptedDataA })
        });

        await fetch("http://127.0.0.1:5000/store_third_party", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ data: encryptedDataB })
        });

        document.getElementById("responseMessage").textContent = "Data successfully sent!";
        document.getElementById("responseMessage").classList.remove("d-none");
    } catch (error) {
        console.error("Error sending data:", error);
        alert("Failed to send data. Please try again.");
    }
}

function encodeGender(gender) {
    switch (gender) {
        case "Male": return 0;
        case "Female": return 1;
        case "Non-binary": return 2;
        default: return -1;
    }
}

function encodeAge(age) {
    switch (age) {
        case "Under 18": return 0;
        case "18-25": return 1;
        case "26-40": return 2;
        case "41-60": return 3;
        case "60+": return 4;
        default: return -1;
    }
}