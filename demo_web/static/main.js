$(function() {

    $('#calc').click(function() {
        $.ajax({
            url : '/api/calc?a=' + document.getElementById('a').value ,
            success: function(data) {
                $('#add').html(data['result']);
            }
        });
    });
})