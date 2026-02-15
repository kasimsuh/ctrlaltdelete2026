import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VHR Spike Demo",
  description: "Record webcam clip, upload to FastAPI, and print open-rppg heart-rate estimates."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
