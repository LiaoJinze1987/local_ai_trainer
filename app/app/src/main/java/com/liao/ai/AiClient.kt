package com.liao.ai

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject

class AiClient {

    private val client = OkHttpClient()

    private val serverUrl = "http://192.168.1.9:8000"

    fun checkStatus(): Boolean {
        val request = Request.Builder()
            .url("$serverUrl/status")
            .get()
            .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw RuntimeException("HTTP ${response.code}")
            }

            val json = JSONObject(response.body?.string() ?: "")
            return json.optBoolean("loaded", false)
        }
    }

    fun chat(
        prompt: String,
        maxNewTokens: Int = 200,
        temperature: Double = 0.7
    ): String {
        val json = JSONObject()
        json.put("prompt", prompt)
        json.put("max_new_tokens", maxNewTokens)
        json.put("temperature", temperature)

        val body = json.toString()
            .toRequestBody("application/json".toMediaType())

        val request = Request.Builder()
            .url("$serverUrl/chat")
            .post(body)
            .build()

        client.newCall(request).execute().use { response ->
            val text = response.body?.string() ?: ""

            if (!response.isSuccessful) {
                val error = JSONObject(text)
                throw RuntimeException(
                    error.optString("detail", "请求失败")
                )
            }

            val result = JSONObject(text)

            return result.optString(
                "response",
                "模型没有返回内容"
            )
        }
    }
}