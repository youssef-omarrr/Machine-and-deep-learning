var animateButton = function(e) {
    e.preventDefault();
    // reset animation
    e.target.classList.remove('animate');
    void e.target.offsetWidth; // trigger reflow for restart
    e.target.classList.add('animate');
    setTimeout(function(){
        e.target.classList.remove('animate');
    },700);
};

var clearButton = document.getElementById("clear");
if (clearButton) {
    clearButton.addEventListener('click', animateButton, false);
}