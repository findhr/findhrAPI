async function encryptData(data) {
    const publicKeyPem = await fetch("../RSA/public_key.pem").then(res => res.text());
    const publicKey = await window.crypto.subtle.importKey(
        "spki",
        base64ToArrayBuffer(publicKeyPem),
        { name: "RSA-OAEP", hash: "SHA-256" },
        false,
        ["encrypt"]
    );

    const encodedData = new TextEncoder().encode(JSON.stringify(data));
    const encryptedData = await window.crypto.subtle.encrypt(
        { name: "RSA-OAEP" },
        publicKey,
        encodedData
    );

    return arrayBufferToBase64(encryptedData);
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
