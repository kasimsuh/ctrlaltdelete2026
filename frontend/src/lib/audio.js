export const createAudioBuffer = (audioContext, pcm16, sampleRate) => {
  const floatData = new Float32Array(pcm16.length);
  for (let i = 0; i < pcm16.length; i += 1) {
    floatData[i] = Math.max(-1, Math.min(1, pcm16[i] / 32768));
  }
  const buffer = audioContext.createBuffer(1, floatData.length, sampleRate);
  buffer.copyToChannel(floatData, 0);
  return buffer;
};

export const decodeBase64ToInt16 = (base64) => {
  const binary = atob(base64);
  const length = binary.length;
  const bytes = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Int16Array(bytes.buffer);
};

export const encodeToBase64 = (int16) => {
  const bytes = new Uint8Array(int16.buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
};

export const resampleTo16k = (input, inputRate) => {
  if (inputRate === 16000) {
    return input;
  }
  const ratio = inputRate / 16000;
  const newLength = Math.round(input.length / ratio);
  const output = new Float32Array(newLength);
  for (let i = 0; i < newLength; i += 1) {
    const position = i * ratio;
    const leftIndex = Math.floor(position);
    const rightIndex = Math.min(leftIndex + 1, input.length - 1);
    const weight = position - leftIndex;
    output[i] = input[leftIndex] * (1 - weight) + input[rightIndex] * weight;
  }
  return output;
};

export const floatToInt16 = (floatData) => {
  const output = new Int16Array(floatData.length);
  for (let i = 0; i < floatData.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, floatData[i]));
    output[i] = sample < 0 ? sample * 32768 : sample * 32767;
  }
  return output;
};
